#import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as mdates
import seaborn as sns

from pandas.core import datetools
import statsmodels.api as sm  
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf  
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")
#read dataset
raw = pd.read_csv('../input/CryptocoinsHistoricalPrices.csv')
raw.head()
#extract ETH close price data from the raw dataset
df = raw[raw['coin']=='ETH'][['Date','Close']]
df['Date'] = pd.to_datetime(df['Date'])
df=df.sort_values(by=['Date'])
#set Date column as index
df = df.set_index('Date')
df
fig1 = plt.figure(figsize=(12,8),dpi=100)
ax1 = fig1.add_subplot(111)
ax1.set_ylabel('Close Price (USD)',fontsize=14, color='black')
ax1.set_xlabel('Date',fontsize=14, color='black')
plt.plot(df)
#decompose the close price curve
#given that 1 years has 255 trading days
decompose = seasonal_decompose(df['Close'], freq=255, model='additive')
fig1 = plt.figure(figsize=(12,8),dpi=100)

ax1 = fig1.add_subplot(411)
ax1.plot(df,label='Raw data')
ax1.legend(loc='upper left')

ax2 = fig1.add_subplot(412)
ax2.plot(decompose.trend,label='Trend')
ax2.legend(loc='upper left')

ax3 = fig1.add_subplot(413)
ax3.plot(decompose.seasonal,label='Seasonal')
ax3.legend(loc='upper left')

ax4 = fig1.add_subplot(414)
ax4.plot(decompose.resid,label='Residual')
ax4.set_xlabel('Date',fontsize=14,color='black')
ax4.legend(loc='upper left')

plt.show()

def diff(data):
    diff = [np.NaN]
    for i in range(1,len(data)):
        if np.isnan(data[i-1]):
            diff.append(np.NaN)
        else:
            diff.append(data[i]-data[i-1])
    return diff
#calculate the differences between close prices
df['dClose'] = diff(df['Close'])
#decompose the differentiated close price curve
#given that 1 years has 255 trading days
decompose_dClose = seasonal_decompose(df['dClose'][1:], freq=255, model='additive')

fig1 = plt.figure(figsize=(12,8),dpi=100)

ax1 = fig1.add_subplot(411)
ax1.plot(df['dClose'][1:],label='dClose')
ax1.legend(loc='upper left')

ax2 = fig1.add_subplot(412)
ax2.plot(decompose_dClose.trend,label='Trend')
ax2.legend(loc='upper left')

ax3 = fig1.add_subplot(413)
ax3.plot(decompose_dClose.seasonal,label='Seasonal')
ax3.legend(loc='upper left')

ax4 = fig1.add_subplot(414)
ax4.plot(decompose_dClose.resid,label='Residual')
ax4.set_xlabel('Date',fontsize=14,color='black')
ax4.legend(loc='upper left')

plt.show()

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
sm.graphics.tsa.plot_acf(df['Close'], lags=255, ax=ax1)

ax2 = fig.add_subplot(212)
sm.graphics.tsa.plot_pacf(df['dClose'][1:], lags=255, ax=ax2)
plt.show()
df['Close'].hist()
df['dClose'][1:].hist()
print('p-values of \'Close\' =',adfuller(df['Close'])[1])
print('p-values of \'dClose\' =',adfuller(df['dClose'][1:])[1])
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
sm.graphics.tsa.plot_acf(df['dClose'][1:], lags=10, ax=ax1)

ax2 = fig.add_subplot(212)
sm.graphics.tsa.plot_pacf(df['dClose'][1:], lags=10, ax=ax2)
plt.show()
model610 = ARIMA(df['Close'], order=(6,1,0))
fit_model610 = model610.fit()
print(fit_model610.summary())
model016 = ARIMA(df['Close'], order=(0,1,6))
fit_model016 = model016.fit()
print(fit_model016.summary())
model616 = ARIMA(df['Close'], order=(6,1,6))
fit_model616 = model616.fit()
print(fit_model616.summary())
size = int((len(df)-1) * 0.90)
train = df['Close']
train['train'] = train[0:size]
train, test = df['Close'][1:size], df['Close'][size:(len(df)-1)]

#error = mean_squared_error(test, predictions)

fig, ax1 = plt.subplots(figsize=(10,8),dpi=100)
fig = fit_model616.plot_predict(start = df.index[1],end = '2018-09', ax=ax1)
ax1.set_ylabel('Close Price (USD)',fontsize=14, color='black')
ax1.set_xlabel('Date',fontsize=14, color='black')

#inset = plt.axes([0.2, 0.35, 0.4, 0.35])
#fig2 = fit_model616.plot_predict(start = '2018-08',end = '2018-09', ax=inset)

legend = ax1.legend(loc='upper left')