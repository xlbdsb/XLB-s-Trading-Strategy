import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import pandas_datareader.data as web
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
import re
from plotly.subplots import make_subplots

# ==========================================
# 0. 页面配置
# ==========================================
st.set_page_config(page_title="TQQQ Trading Monitor", layout="wide", page_icon="📈")

st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
        text-align: center;
        margin-bottom: 10px;
    }
    .signal-buy { color: #00e676; font-weight: bold; font-size: 24px; }
    .signal-sell { color: #ff1744; font-weight: bold; font-size: 24px; }
    .stButton>button { width: 100%; font-weight: bold; }
    /* Metric styling adjustment for better visibility */
    div[data-testid="stMetric"] {
        background-color: #262730;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 核心引擎 (CoreEngine)
# ==========================================
class CoreEngine:
    def __init__(self):
        self.tickers = {
            'QQQ': 'QQQ', 'SPY': 'SPY',
            'VIX': '^VIX', 'JNK': 'JNK', 'LQD': 'LQD',
            'TQQQ': 'TQQQ', 'UPRO': 'UPRO'
        }
        self.fred_tickers = {
            'Credit_Spread': 'BAMLH0A0HYM2',
            'Fed_Funds': 'FEDFUNDS',
            'Fed_Assets': 'WALCL',
            'TGA': 'WTREGEN',
            'RRP': 'RRPONTSYD'
        }

    def fetch_data(self, start_date, end_date):
        """Unified Data Fetching"""
        # 1. Yahoo Finance
        try:
            fetch_start = start_date - datetime.timedelta(days=730)
            yf_data = yf.download(list(self.tickers.values()), start=fetch_start, end=end_date, progress=False, auto_adjust=True)
            
            if isinstance(yf_data.columns, pd.MultiIndex):
                try: df = yf_data.xs('Close', axis=1, level=0, drop_level=True)
                except: df = yf_data['Close']
            else:
                df = yf_data['Close']
            
            inv_map = {v: k for k, v in self.tickers.items()}
            df = df.rename(columns=lambda x: inv_map.get(x, x))
            
        except Exception as e:
            st.error(f"Yahoo Data Error: {e}")
            return pd.DataFrame()

        # 2. FRED Data
        try:
            fred = web.DataReader(list(self.fred_tickers.values()), 'fred', fetch_start, end_date)
            fred = fred.resample('D').ffill()
            fred.columns = list(self.fred_tickers.keys())
            df = df.join(fred).ffill()
        except Exception as e:
            st.warning(f"FRED Data Warning: {e}")
            for col in self.fred_tickers.keys():
                df[col] = np.nan

        return df.dropna(subset=['QQQ']) 

    def calculate_indicators(self, df):
        df = df.copy()
        
        if 'JNK' in df.columns:
            df['JNK_Inverse'] = 1 / df['JNK'] * 100
        
        for asset in ['QQQ', 'SPY']:
            if asset in df.columns:
                df[f'{asset}_MA50'] = df[asset].rolling(50).mean()
                df[f'{asset}_MA100'] = df[asset].rolling(100).mean()
                df[f'{asset}_MA200'] = df[asset].rolling(200).mean()
                df[f'{asset}_EMA20'] = df[asset].ewm(span=20, adjust=False).mean()

        df['Spread'] = df['Credit_Spread']
        df['Spread_MA20'] = df['Spread'].rolling(20).mean()
        df['Spread_MA200'] = df['Spread'].rolling(200).mean()
        df['Spread_Low_200'] = df['Spread'].rolling(200).min()
        df['Spread_High_200'] = df['Spread'].rolling(200).max()
        df['Spread_Peak_10'] = df['Spread'].rolling(10).max()
        
        df['JNK_Spread_MA200'] = df['JNK_Inverse'].rolling(200).mean()
        df['JNK_Spread_Low_200'] = df['JNK_Inverse'].rolling(200).min()
        
        if 'Fed_Assets' in df.columns:
            df['Net_Liquidity'] = df['Fed_Assets'] - df['TGA'] - df['RRP']
        
        df['H_EMA'] = df['Spread'].ewm(span=220).mean()
        df['H_Sell_Trigger'] = df['Spread_Low_200'] * 1.33
        
        df['Rate_Chg'] = df['Fed_Funds'].diff(63)
        df['Fed_Cycle'] = np.where(df['Rate_Chg'] > 0.1, 'Hiking', 
                          np.where(df['Rate_Chg'] < -0.1, 'Cutting', 'Neutral'))
        
        if 'SPY' in df.columns and 'VIX' in df.columns:
            spy_ma125 = df['SPY'].rolling(125).mean()
            mom_diff = (df['SPY'] - spy_ma125) / spy_ma125
            score_mom = (50 + (mom_diff * 500)).clip(0, 100)
            score_vix = (100 - (df['VIX'] - 12) * 3.33).clip(0, 100)
            df['Fear_Greed_Syn'] = (score_mom * 0.6 + score_vix * 0.4)
        else:
            df['Fear_Greed_Syn'] = 50 
            
        return df

    def apply_strategies(self, df, asset='QQQ'):
        price = df[asset]
        ma200 = df[f'{asset}_MA200']
        ema20 = df[f'{asset}_EMA20']
        spread = df['Spread']
        
        sig_h = []
        curr_h = 0.0
        for i in range(len(df)):
            row = df.iloc[i]
            s = row['Spread']
            h_sell = (s > row['H_Sell_Trigger']) and (s > row['H_EMA'])
            if curr_h == 0.0:
                if not h_sell: curr_h = 1.0 
            else:
                if h_sell: curr_h = 0.0
                else: curr_h = 1.0
            sig_h.append(curr_h)
        df['Signal_H'] = sig_h

        df['Signal_I'] = np.where(price > ma200, 1.0, 0.0)

        sig_n = []
        curr_n = 0.0
        n_macro_block = False 
        for i in range(len(df)):
            row = df.iloc[i]
            p = row[asset]
            s = row['Spread']
            s_ma200 = row['Spread_MA200']
            
            if n_macro_block:
                if s < s_ma200 or s < (row['Spread_High_200'] * 0.70):
                    n_macro_block = False
            else:
                is_rising = s >= row['Spread_Peak_10'] * 0.99
                if (s > row['Spread_Low_200'] * 1.33) and (s > s_ma200) and is_rising:
                    n_macro_block = True
            
            new_n = curr_n
            if curr_n == 1.0:
                if (p < row[f'{asset}_MA200']) or n_macro_block:
                    new_n = 0.0
            else:
                if not n_macro_block:
                    if row['Fed_Cycle'] == 'Cutting':
                        if p > row[f'{asset}_EMA20'] and s < row['Spread_MA20']:
                            new_n = 1.0
                    else:
                        if p > row[f'{asset}_MA200']:
                            new_n = 1.0
            curr_n = new_n
            sig_n.append(curr_n)
        
        df['Signal_N'] = sig_n
        return df
    
    def get_sentiment_index(self):
        try:
            url = "https://feargreedmeter.com/"
            headers = {'User-Agent': 'Mozilla/5.0'}
            r = requests.get(url, headers=headers, timeout=5)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, 'html.parser')
                text = soup.get_text()
                match = re.search(r'(?:Now|Current|Index):\s*(\d+)', text, re.IGNORECASE)
                if not match:
                    match = re.search(r'Fear.*?(\d{1,3}).*?Greed', text, re.DOTALL)
                if match:
                    val = int(match.group(1))
                    if 0 <= val <= 100: return val, "FearGreedMeter.com"
        except: pass

        try:
            url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
            headers = {"User-Agent": "Mozilla/5.0"}
            r = requests.get(url, headers=headers, timeout=5)
            if r.status_code == 200:
                data = r.json()
                val = int(data['fear_and_greed_historical_data'][0]['score'])
                return val, "CNN Official API"
        except: pass
            
        return None, "Synthetic (Fallback)"

# ==========================================
# 2. 绘图模块
# ==========================================
def get_xaxis_config(min_date, max_date):
    return dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1Y", step="year", stepmode="backward"), 
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=3, label="3Y", step="year", stepmode="backward"),
                dict(count=5, label="5Y", step="year", stepmode="backward"),
                dict(step="all", label="MAX")
            ]),
            bgcolor="#1e1e1e",
            font=dict(color="white")
        ),
        type="date",
        range=[min_date, max_date]
    )

def plot_price_monitor(df, asset):
    last_price = df[asset].iloc[-1]
    ma200 = df[f'{asset}_MA200'].iloc[-1]
    
    max_d = df.index[-1]
    min_d = max_d - pd.DateOffset(years=1)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[asset], name=f'{asset}', line=dict(color='white', width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df[f'{asset}_MA50'], name='MA50', line=dict(color='cyan', width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df[f'{asset}_MA100'], name='MA100', line=dict(color='yellow', width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df[f'{asset}_MA200'], name='MA200', line=dict(color='orange', width=2)))
    
    color = '#00e676' if last_price > ma200 else '#ff1744'
    status = "BULLISH" if last_price > ma200 else "BEARISH"
    
    fig.update_layout(
        title=f"{asset} Trend (<span style='color:{color}'>{status}</span>)",
        template="plotly_dark", height=500, hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=get_xaxis_config(min_d, max_d),
        yaxis=dict(fixedrange=False)
    )
    return fig

def plot_spread_monitor(df, col, ma, low, title):
    curr = df[col].iloc[-1]
    thresh = df[low].iloc[-1] * 1.30
    
    max_d = df.index[-1]
    min_d = max_d - pd.DateOffset(years=1)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[col], name='Spread', line=dict(color='#00e676', width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df[ma], name='MA200', line=dict(color='gray', dash='dot')))
    fig.add_trace(go.Scatter(x=df.index, y=df[low]*1.30, name='Algo Threshold', line=dict(color='red', dash='dash')))
    
    if 'FRED' in title:
        fig.add_hrect(y0=8.0, y1=20.0, fillcolor="red", opacity=0.1, line_width=0, annotation_text="CRISIS (>8%)", annotation_position="top left")
        fig.add_hrect(y0=5.5, y1=8.0, fillcolor="orange", opacity=0.1, line_width=0, annotation_text="ELEVATED (5.5-8%)", annotation_position="top left")
        fig.add_hrect(y0=3.5, y1=5.5, fillcolor="green", opacity=0.1, line_width=0, annotation_text="NORMAL (3.5-5.5%)", annotation_position="top left")
        fig.add_hrect(y0=0.0, y1=3.5, fillcolor="blue", opacity=0.1, line_width=0, annotation_text="TIGHT (<3.5%)", annotation_position="top left")

    status = "DANGER" if curr > thresh else "NORMAL"
    color = '#ff1744' if curr > thresh else '#00e676'
    
    fig.update_layout(
        title=f"{title} (<span style='color:{color}'>{status}</span>)",
        template="plotly_dark", height=400, hovermode="x unified", 
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=get_xaxis_config(min_d, max_d),
        yaxis=dict(fixedrange=False)
    )
    return fig

def plot_liquidity(df):
    max_d = df.index[-1]
    min_d = max_d - pd.DateOffset(years=1)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df.index, y=df['SPY'], name='S&P 500', line=dict(color='gray', width=1, dash='dot')), secondary_y=False)
    
    curr_liq = df['Net_Liquidity'].iloc[-1]
    prev_liq = df['Net_Liquidity'].iloc[-20]
    color = '#00e676' if curr_liq > prev_liq else '#ff1744'
    status = "EXPANDING" if curr_liq > prev_liq else "CONTRACTING"
    
    fig.add_trace(go.Scatter(x=df.index, y=df['Net_Liquidity'], name='Net Liquidity', 
                             line=dict(color='#29b6f6', width=2), fill='tozeroy', fillcolor='rgba(41, 182, 246, 0.1)'), secondary_y=True)
    
    fig.update_layout(
        title=f"Fed Net Liquidity (<span style='color:{color}'>{status}</span>)",
        template="plotly_dark", height=450, hovermode="x unified", margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", y=1.1),
        xaxis=get_xaxis_config(min_d, max_d),
        yaxis=dict(fixedrange=False),
        yaxis2=dict(fixedrange=False)
    )
    fig.update_yaxes(title_text="Net Liquidity ($M)", secondary_y=True, showgrid=False)
    return fig

def plot_gauge(val, source):
    if val is None: val = 50
    if val >= 75: color = "#00e676"; text = "Extreme Greed"
    elif val >= 55: color = "#69f0ae"; text = "Greed"
    elif val >= 45: color = "#bdbdbd"; text = "Neutral"
    elif val >= 25: color = "#ff5252"; text = "Fear"
    else: color = "#d50000"; text = "Extreme Fear"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number", value = val,
        title = {'text': f"<b>{text}</b><br><span style='font-size:14px;color:gray'>Src: {source}</span>", 'font': {'size': 24, 'color': color}},
        gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "white"},
                 'steps': [{'range': [0, 25], 'color': '#d50000'}, {'range': [75, 100], 'color': '#00e676'}]}
    ))
    fig.update_layout(paper_bgcolor="#1e1e1e", font={'color': "white"}, height=250, margin=dict(t=50, b=20, l=30, r=30))
    return fig

# ==========================================
# 3. Main Logic
# ==========================================
core = CoreEngine()

st.sidebar.title("TQQQ Trading Monitor")
page = st.sidebar.radio("Navigation", ["🚀 市场雷达 (Dashboard)", "🛡️ 完整回测 (Full Backtest)"])

if page == "🚀 市场雷达 (Dashboard)":
    st.title("🚀 TQQQ Trading Monitor")
    
    if st.button("🔄 刷新数据"): st.cache_data.clear()
    
    # 获取明天日期以确保包含今天的最新数据
    end_date = datetime.date.today() + datetime.timedelta(days=1)
    start_date = datetime.date(2000, 1, 1) 
    
    with st.spinner("正在同步数据 (Yahoo + FRED + Sentiment)..."):
        df = core.fetch_data(start_date, end_date)
        fg_val, fg_src = core.get_sentiment_index()
        
        if df.empty:
            st.error("数据获取失败，请检查网络连接。")
        else:
            df = core.calculate_indicators(df)
            
            # --- 0. 市场概览 (Market Overview) ---
            st.subheader("📊 市场概览 (Market Today)")
            latest_date = df.index[-1].strftime('%Y-%m-%d')
            st.caption(f"Data as of: {latest_date}")
            
            m1, m2, m3, m4 = st.columns(4)
            
            # Helper for metrics
            def show_metric(col, label, current, prev, is_inverse=False):
                diff = current - prev
                pct = (diff / prev) * 100
                color = "normal"
                if is_inverse: color = "inverse"
                col.metric(label, f"{current:.2f}", f"{diff:.2f} ({pct:.2f}%)", delta_color=color)

            # QQQ & SPY
            if len(df) >= 2:
                show_metric(m1, "QQQ", df['QQQ'].iloc[-1], df['QQQ'].iloc[-2])
                show_metric(m2, "SPY", df['SPY'].iloc[-1], df['SPY'].iloc[-2])
                
                # VIX (Inverse color: down is good)
                if 'VIX' in df.columns:
                    show_metric(m3, "VIX", df['VIX'].iloc[-1], df['VIX'].iloc[-2], is_inverse=True)
            
            # Fear & Greed
            if fg_val:
                m4.metric("Fear & Greed", f"{int(fg_val)}", f"{fg_src}")

            st.markdown("---")

            # --- 1. 顶部信号栏 ---
            st.subheader("📡 核心策略信号 (Real-time)")
            asset_view = st.radio("基准资产:", ["QQQ", "SPY"], horizontal=True)
            
            df_sig = core.apply_strategies(df, asset_view)
            last = df_sig.iloc[-1]
            
            h_sig = "BUY" if last['Signal_H'] == 1 else "SELL"
            i_sig = "BUY" if last['Signal_I'] == 1 else "SELL"
            n_sig = "BUY" if last['Signal_N'] == 1 else "SELL"
            
            n_reason = "Safe"
            if n_sig == "SELL":
                if last[asset_view] < last[f'{asset_view}_MA200']: n_reason = "Price < MA200"
                else: n_reason = "Macro Blowout"
            
            c1, c2, c3 = st.columns(3)
            def card(c, t, s, d):
                color = "signal-buy" if s == "BUY" else "signal-sell"
                c.markdown(f'<div class="metric-card"><h4>{t}</h4><div class="{color}">{s}</div><small>{d}</small></div>', unsafe_allow_html=True)
            
            with c1: card(c1, "Strategy H", h_sig, "Macro Cycle")
            with c2: card(c2, "Strategy I", i_sig, "Trend Following")
            with c3: card(c3, "Strategy N", n_sig, n_reason)
            
            st.markdown("---")
            
            # --- 2. 价格趋势 ---
            st.subheader(f"📈 价格趋势 ({asset_view})")
            st.plotly_chart(plot_price_monitor(df, asset_view), use_container_width=True)
            
            # --- 3. 宏观风险 ---
            st.subheader("🌊 宏观风险与流动性")
            
            m1, m2 = st.columns(2)
            with m1:
                st.plotly_chart(plot_spread_monitor(df, 'Spread', 'Spread_MA200', 'Spread_Low_200', 'Credit Spread (FRED)'), use_container_width=True)
            with m2:
                st.plotly_chart(plot_spread_monitor(df, 'JNK_Inverse', 'JNK_Spread_MA200', 'JNK_Spread_Low_200', 'Real-time Proxy (1/JNK)'), use_container_width=True)
                
            # --- 4. 净流动性 ---
            st.subheader("🏦 美联储净流动性 (Don't Fight the Fed)")
            st.plotly_chart(plot_liquidity(df), use_container_width=True)
            
            # --- 5. 情绪 ---
            st.subheader("😨 市场情绪")
            s1, s2 = st.columns([1, 2])
            with s1:
                if fg_val:
                    st.plotly_chart(plot_gauge(fg_val, fg_src), use_container_width=True)
                else:
                    syn_val = df['Fear_Greed_Syn'].iloc[-1]
                    st.plotly_chart(plot_gauge(syn_val, "Synthetic (All API Failed)"), use_container_width=True)
            with s2:
                if 'VIX' in df.columns:
                    fig_vix = go.Figure()
                    fig_vix.add_trace(go.Scatter(x=df.index, y=df['VIX'], name='VIX', line=dict(color='#e040fb', width=2)))
                    
                    max_d = df.index[-1]
                    min_d = max_d - pd.DateOffset(years=1)
                    
                    fig_vix.update_layout(
                        title="VIX Index", 
                        template="plotly_dark", 
                        height=250, 
                        xaxis=get_xaxis_config(min_d, max_d),
                        yaxis=dict(fixedrange=False)
                    )
                    st.plotly_chart(fig_vix, use_container_width=True)

elif page == "🛡️ 完整回测 (Full Backtest)":
    st.header("🛡️ 完整策略回测")
    
    col1, col2, col3 = st.columns(3)
    asset_select = col1.selectbox("Asset", ["Nasdaq-100 (QQQ)", "S&P 500 (SPY)"])
    today = datetime.date.today()
    start_d = col2.date_input("Start", datetime.date(2008, 1, 1), max_value=today)
    end_d = col3.date_input("End", today, max_value=today)
    
    if st.button("🚀 运行完整回测"):
        asset_code = "QQQ" if "QQQ" in asset_select else "SPY"
        
        with st.spinner("Calculating..."):
            df_bt = core.fetch_data(start_d, end_d)
            
            if not df_bt.empty:
                df_bt = core.calculate_indicators(df_bt)
                df_bt = core.apply_strategies(df_bt, asset_code)
                
                daily_cost = 0.00035 
                ret_3x = df_bt[asset_code].pct_change() * 3 - daily_cost
                
                metrics = []
                def calc_dd(series):
                    peak = series.cummax()
                    return ((series - peak) / peak).min()

                eq_bh = 10000 * (1 + ret_3x).cumprod()
                metrics.append({"Strategy": "Buy & Hold (3x)", "Final": eq_bh.iloc[-1], "MaxDD": calc_dd(eq_bh)})
                
                plot_data = pd.DataFrame(index=df_bt.index)
                plot_data['Buy & Hold'] = eq_bh
                
                for s in ['H', 'I', 'N']:
                    sig = df_bt[f'Signal_{s}'].shift(1).fillna(0)
                    eq = 10000 * (1 + ret_3x * sig).cumprod()
                    metrics.append({"Strategy": f"Strategy {s}", "Final": eq.iloc[-1], "MaxDD": calc_dd(eq)})
                    plot_data[f'Strategy {s}'] = eq
                
                res = pd.DataFrame(metrics).sort_values('Final', ascending=False)
                res['Final'] = res['Final'].map('${:,.0f}'.format)
                res['MaxDD'] = res['MaxDD'].map('{:.2%}'.format)
                st.dataframe(res, use_container_width=True)
                
                fig_eq = go.Figure()
                for c in plot_data.columns:
                    fig_eq.add_trace(go.Scatter(x=plot_data.index, y=plot_data[c], name=c))
                
                max_bd = df_bt.index[-1]
                min_bd = max_bd - pd.DateOffset(years=1)
                
                fig_eq.update_yaxes(type="log", fixedrange=False)
                fig_eq.update_layout(template="plotly_dark", height=500, xaxis=get_xaxis_config(min_bd, max_bd))
                st.plotly_chart(fig_eq, use_container_width=True)
                
            else:
                st.error("No Data Found.")