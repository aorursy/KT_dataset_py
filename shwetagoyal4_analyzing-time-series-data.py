import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as msno



from plotly.offline import init_notebook_mode, iplot



import plotly.offline as py

import plotly.graph_objs as go

import plotly.express as px

py.init_notebook_mode(connected=True)



import warnings

warnings.filterwarnings('ignore')



# Few libraries for visualizing time series data

from pylab import rcParams

from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
data = pd.read_csv("../input/nifty50-stock-market-data/COALINDIA.csv")

data.set_index("Date", drop=False, inplace=True)
data.head()
# check missing/null values



msno.matrix(data)
fig = px.line(data, x="Date", y="VWAP")

py.iplot(fig, filename="simple_line")
# Let's take CoalIndia stocks High price for this

data1 = pd.read_csv("../input/nifty50-stock-market-data/COALINDIA.csv", index_col="Date", parse_dates=["Date"])



Plot = seasonal_decompose(data1["High"], freq=360)

plt.rcParams.update({'figure.figsize' : (10, 10)})

Plot.plot()

plt.show()
fig = go.Figure(data=[go.Ohlc(

    x=data.index,

    open=data.Open, high=data.High,

    low=data.Low, close=data.Close,

    increasing_line_color= 'cyan', decreasing_line_color= 'gray'

)])

py.iplot(fig, filename="simple_ohlc")
fig = go.Figure(data=[go.Candlestick(

    x=data.index,

    open=data.Open, high=data.High,

    low=data.Low, close=data.Close,)])

py.iplot(fig, filename="simple_candlestick")
plot_acf(data.VWAP, lags=50, title="VWAP")

plt.show()
plot_pacf(data.VWAP, lags=50, title="VWAP")

plt.show()