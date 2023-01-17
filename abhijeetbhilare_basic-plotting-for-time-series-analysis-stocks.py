!pip install dabl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import missingno as msno
from scipy import stats
sns.set(color_codes=True)
import warnings
warnings.filterwarnings('ignore')
from plotly.offline import init_notebook_mode, iplot

import plotly.offline as py
import plotly.graph_objs as go
import plotly.express as px
py.init_notebook_mode(connected=True)
from dabl import plot
df = pd.read_csv("/kaggle/input/nifty50-stock-market-data/NIFTY50_all.csv", parse_dates=['Date'])
df.head()
df.isna().any()
def plot_df(df, x, y, title="", xlabel='Stock name', ylabel='Value', dpi=100):
    plt.figure(figsize=(80,16), dpi=dpi)
    plt.xticks(rotation=90, ha='right')
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()
plot_df(df, x=df.Symbol, y=df.VWAP, title='Nifty50')  
plot_df(df, x=df.Symbol, y=df.Turnover, title='Nifty50')  
msno.matrix(df)
sns.pairplot(df)
fig = px.line(df, x="Date", y="VWAP")
py.iplot(fig, filename="simple_line")
fig = px.line(df, x="Symbol", y="Trades")
py.iplot(fig, filename="simple_line")
plot(df, "VWAP")