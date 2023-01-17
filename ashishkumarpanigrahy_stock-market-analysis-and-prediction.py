# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
Q = pd.read_csv('../input/stock-market-wide-datasets/000000000000Q',header=0, parse_dates=[0], index_col=0, squeeze=True)

AM = pd.read_csv('../input/stock-market-wide-datasets/AM',header=0, parse_dates=[0], index_col=0, squeeze=True)

event = pd.read_csv('../input/stock-market-wide-datasets/event',header=0, parse_dates=[0], index_col=0, squeeze=True)

news = pd.read_csv('../input/stock-market-wide-datasets/news',header=0, parse_dates=[0], index_col=0, squeeze=True)
Q.head()
Q.tail()
AM.head()
AM.tail()
event.head()
event.tail()
news.head()
Q.info()
AM.describe()
Q.ticker.value_counts()[:3]
AM.symbol.value_counts()[:3]
plt.figure(figsize=(10,10))

plt.bar(Q.ticker.value_counts().index[:20],Q.ticker.value_counts()[:20])

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(15,10))

plt.bar(AM.symbol.value_counts().index[:101],AM.symbol.value_counts()[:101])

plt.xticks(rotation=90)

plt.show()
AM.symbol[AM.symbol == 'T']
plt.figure(figsize=(10,5))

plt.scatter(range(len(AM.symbol[AM.symbol == 'T'][:1000])),AM.symbol[AM.symbol == 'T'].index[:1000])

plt.ylabel('time')

plt.xticks(rotation=60)
plt.figure(figsize=(15,5))

plt.plot(AM.symbol[AM.symbol=='T'].index)

plt.show()
plt.figure(figsize=(15,5))

plt.plot(Q.ticker[Q.ticker=='T'].index, c='r')

plt.show()
plt.figure(figsize=(20,5))

AM_raw = pd.read_csv('../input/stock-market-wide-datasets/AM')

Q_raw = pd.read_csv('../input/stock-market-wide-datasets/000000000000Q')

plt.plot(Q_raw[Q_raw.ticker == 'T'][:50000].bid_price, label='bid_price')

plt.plot(Q_raw[Q_raw.ticker == 'T'][:50000].ask_price,c='r', label='ask_price')

plt.plot(AM_raw[AM_raw.symbol == 'T'][:50000].close_price, c='g',label='close_price')

plt.legend(loc='best')
AM_raw.columns
plt.figure(figsize=(20,5))

plt.plot(AM_raw.volume[:100])

plt.plot(AM_raw.accumulated_volume[:100], c= 'g')

plt.plot(AM_raw.volume[:100].cumsum(), c='r')
plt.figure(figsize=(20,5))

plt.plot(AM_raw.open_price[:1000], label='open price')

plt.plot(AM_raw.high_price[:1000], c='g',label='high price')

plt.plot(AM_raw.low_price[:1000], c='r', label='low price')

plt.plot(AM_raw.average_price[:1000], c='orange',label='average price')

#plt.plot(AM_raw.close_price, c='k',label='close price')

plt.legend(loc='best')

plt.show()
plt.figure(figsize=(20,5))

plt.plot(AM_raw.close_price[:1000], c='k',label='close price')
plt.figure(figsize=(20,5))

plt.plot(AM_raw.VWAP[:1000])
plt.figure(figsize=(20,5))

plt.scatter(AM.index.minute[:10000],AM.VWAP[:10000])
import plotly.express as px

px.scatter(Q, x='ask_price',y='bid_price',color='ticker', title='Ask Price vs Bid price')
from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.arima_model import ARIMA

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
plt.plot(AM.close_price)
mini_AM = AM[:1000]
rolling_mean = mini_AM.close_price.rolling(window = 12).mean()

rolling_std = mini_AM.close_price.rolling(window = 12).std()



plt.plot(mini_AM.close_price, color = 'green', label = 'Original')

plt.plot(rolling_mean, color = 'red', label = 'Rolling Mean')

plt.plot(rolling_std, color = 'black', label = 'Rolling Std')

plt.legend(loc = 'best')

plt.title('Rolling Mean & Rolling Standard Deviation')

plt.show()
result = adfuller(mini_AM['close_price'])

print('ADF Statistic: {}'.format(result[0]))

print('p-value: {}'.format(result[1]))

print('Critical Values:')

for key, value in result[4].items():

    print('\t{}: {}'.format(key, value))
df_log = np.log(mini_AM.close_price)

plt.plot(df_log)
def get_stationarity(timeseries):

    

    # rolling statistics

    rolling_mean = timeseries.rolling(window=12).mean()

    rolling_std = timeseries.rolling(window=12).std()

    

    # rolling statistics plot

    original = plt.plot(timeseries, color='blue', label='Original')

    mean = plt.plot(rolling_mean, color='red', label='Rolling Mean')

    std = plt.plot(rolling_std, color='black', label='Rolling Std')

    plt.legend(loc='best')

    plt.title('Rolling Mean & Standard Deviation')

    plt.show(block=False)

    

    # Dickey–Fuller test:

    result = adfuller(timeseries)

    print('ADF Statistic: {}'.format(result[0]))

    print('p-value: {}'.format(result[1]))

    print('Critical Values:')

    for key, value in result[4].items():

        print('\t{}: {}'.format(key, value))
rolling_mean = df_log.rolling(window=12).mean()

df_log_minus_mean = df_log - rolling_mean

df_log_minus_mean.dropna(inplace=True)

get_stationarity(df_log_minus_mean)
df_log_shift = df_log - df_log.shift()

df_log_shift.dropna(inplace=True)

get_stationarity(df_log_shift)
#decomposition = seasonal_decompose(np.log(mini_AM.close_price)) 

model = ARIMA(df_log, order=(2,1,2))

results = model.fit(disp=-1)

plt.plot(df_log_shift)

plt.plot(results.fittedvalues, color='red')
predictions_ARIMA_diff = pd.Series(results.fittedvalues, copy=True)

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()

predictions_ARIMA_log = pd.Series(df_log.iloc[0], index=df_log.index)

predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)

predictions_ARIMA = np.exp(predictions_ARIMA_log)

plt.scatter(range(len(mini_AM.close_price[:1000])),mini_AM.close_price[:1000], c='r')

plt.scatter(range(len(predictions_ARIMA[:1000])),predictions_ARIMA[:1000],)
results.plot_predict(1,264)