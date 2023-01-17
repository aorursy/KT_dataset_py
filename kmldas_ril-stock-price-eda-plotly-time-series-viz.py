import os



import math

import pandas_datareader as web

import numpy as np

import pandas as pd

from datetime import datetime

import math





from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential

from keras.layers import Dense, LSTM

import matplotlib

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')



import warnings

warnings.filterwarnings('ignore')





import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight') 

%matplotlib inline

from pylab import rcParams

from plotly import tools



from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.figure_factory as ff



import statsmodels.api as sm

from numpy.random import normal, seed

from scipy.stats import norm

from statsmodels.tsa.arima_model import ARMA

from statsmodels.tsa.stattools import adfuller

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.arima_process import ArmaProcess

from statsmodels.tsa.arima_model import ARIMA

import math

from sklearn.metrics import mean_squared_error







ril_price= pd.read_csv("../input/reliance-industries-ril-share-price-19962020/Reliance Industries 1996 to 2020.csv")

#Show the data 

ril_price
ril_price=ril_price.dropna()

ril_price
ril_price.info()
ril_price["Date"]=pd.to_datetime(ril_price["Date"], format="%d-%m-%Y")





ril_price["Date"]



ril_price.set_index('Date', inplace=True)

ril_price.info()
ril_price.describe()
#Visualize the closing price history

plt.figure(figsize=(16,8))

plt.title('Reliance Industries Close Price History')

plt.plot(ril_price['Close'])

plt.xlabel('Date',fontsize=18)

plt.ylabel('Close Price INR',fontsize=18)

plt.show()
ril_price['2012':'2020'].plot(subplots=True, figsize=(10,12))



plt.savefig('stocks12_20.png')

plt.show()
ril_price['2019':'2020'].plot(subplots=True, figsize=(10,12))



plt.savefig('stocks19_20.png')

plt.show()
trace = go.Ohlc(x=ril_price['2020'].index,

                open=ril_price['2020'].Open,

                high=ril_price['2020'].High,

                low=ril_price['2020'].Low,

                close=ril_price['2020'].Close)

data = [trace]



iplot(data, filename='simple_ohlc')
trace = go.Ohlc(x=ril_price['06-2020'].index,

                open=ril_price['06-2020'].Open,

                high=ril_price['06-2020'].High,

                low=ril_price['06-2020'].Low,

                close=ril_price['06-2020'].Close)

data = [trace]



iplot(data, filename='simple_ohlc')
# Candlestick chart of June 2020

trace = go.Candlestick(x=ril_price['06-2020'].index,

                open=ril_price['06-2020'].Open,

                high=ril_price['06-2020'].High,

                low=ril_price['06-2020'].Low,

                close=ril_price['06-2020'].Close)

data = [trace]

iplot(data, filename='simple_candlestick')
ril_price["Close"].plot(figsize=(16,8))
# Now, for decomposition...

rcParams['figure.figsize'] = 11, 9

decomposed_ril_values = sm.tsa.seasonal_decompose(ril_price["Close"],freq=360) # The frequncy is annual

figure = decomposed_ril_values.plot()

plt.show()