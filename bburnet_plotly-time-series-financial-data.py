#import libraries
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt


import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected = True)

import plotly

import datetime

import warnings
warnings.filterwarnings('ignore')
from pylab import rcParams
%matplotlib inline
df_aapl = pd.read_csv('/kaggle/input/sandp500/individual_stocks_5yr/individual_stocks_5yr/AAPL_data.csv')
df_aapl.columns
df_aapl.head()
trace_high = go.Scatter(
    x = df_aapl.date,
    y = df_aapl['high'],
    name = "AAPL Price per Share, highs",
    line = dict(color = '#920a4e'),
    opacity = 0.8)

data = [trace_high]

layout = dict(
    title = 'Time Series',
    xaxis = dict(
        rangeselector = dict(
            buttons = list([
                dict(count = 1,
                     label = '1m',
                     step = 'month',
                     stepmode = 'backward'),
                dict(count = 6,
                     label = '6m',
                     step = 'month',
                     stepmode = 'backward'),
                dict(step = 'all')
            ])
        ),
        rangeslider = dict(
            visible = True
        ),
        type = 'date'
    )
)

fig = dict(data = data, layout = layout)
iplot(fig, filename = "Time Series")
plotly.offline.plot(fig, filename = 'Plotly_AAPL_Stock.html')
fig = go.Figure(data=[go.Candlestick(x = df_aapl['date'],
                                     open = df_aapl['open'],
                                     high = df_aapl['high'],
                                     low = df_aapl['low'],
                                     close = df_aapl['close'])])
fig.update_layout(
    title = 'AAPL Stock, Candlesticks',
    yaxis_title = 'AAPL Stock Price',
    xaxis_title = 'Date',
    template = "plotly_dark")
iplot(fig, filename = "Candlesticks")
plotly.offline.plot(fig, filename = 'Plotly_AAPL_Stock_Candlesticks.html')
df_all = pd.read_csv('/kaggle/input/sandp500/all_stocks_5yr.csv')
df_all.shape
df_all.columns
len(df_all['Name'].unique())
df_sample = df_all[(df_all.Name == 'XOM') 
                   | (df_all.Name == 'NEM') 
                   | (df_all.Name == 'URI') 
                   | (df_all.Name == 'MGM') 
                   | (df_all.Name == 'KO') 
                   | (df_all.Name == 'REGN') 
                   | (df_all.Name == 'JPM')
                   | (df_all.Name == 'NVDA')
                   | (df_all.Name == 'CHTR')
                   | (df_all.Name == 'NEE')
                   | (df_all.Name == 'EQIX')]
ind_dict = {'XOM':'Energy', 'NEM':'Materials',
              'URI':'Industrials', 'MGM':'Consumer Discretionary',
              'KO':'Consumer Staples', 'REGN':'Health Care', 'JPM':'Financials',
              'NVDA':'Information Technology', 'CHTR':'Telecommunication Services',
              'NEE':'Utilities', 'EQIX':'Real Estate'}
df_sample['Industry'] = df_sample['Name'].map(ind_dict)
import plotly.express as px

fig = px.line(df_sample, 
             x = 'date', 
             y = 'close',
             title = 'Time Series Sector Leading Stocks',
             color = 'Industry',
             hover_name = 'Name')
fig.update_xaxes(
    rangeslider_visible = True,
    rangeselector = dict(
        buttons = list([
            dict(count = 1, 
                 label = "1m", 
                 step = "month", 
                 stepmode = "backward"),
            dict(count = 6, 
                 label = "6m", 
                 step = "month", 
                 stepmode = "backward"),
            dict(count = 1, 
                 label = "YTD", 
                 step = "year", 
                 stepmode = "todate"),
            dict(count = 1, 
                 label = "1y", 
                 step = "year", 
                 stepmode = "backward"),
            dict(step = "all")
        ])
    )
)

iplot(fig, filename = "Stocks Sample")
plotly.offline.plot(fig, filename = 'Plotly_Sample_Stocks_Time_Series')