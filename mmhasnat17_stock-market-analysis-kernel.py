import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import pandas_datareader

import datetime

import pandas_datareader.data as web
#start = datetime.datetime(2015, 1, 1)

#end = datetime.datetime(2020, 1, 1)
#The Yahoo API is restricted for some reason from the Kaggle Cloud platform so we are using the Kaggle Cloud dataset 

#apple = web.DataReader("AAPL", 'yahoo',start, end)

#amazon = web.DataReader("AMZN", 'yahoo', start, end)

#google = web.DataReader("GOOGL", 'yahoo', start, end)
apple = pd.read_csv('../input/stock-time-series-20050101-to-20171231/AAPL_2006-01-01_to_2018-01-01.csv', index_col='Date', parse_dates=['Date'])

google = pd.read_csv('../input/stock-time-series-20050101-to-20171231/GOOGL_2006-01-01_to_2018-01-01.csv', index_col='Date', parse_dates=['Date'])

amazon = pd.read_csv('../input/stock-time-series-20050101-to-20171231/AMZN_2006-01-01_to_2018-01-01.csv', index_col='Date', parse_dates=['Date'])
apple.tail()
amazon.tail()
google.head()
apple['Open'].plot(label='Apple',figsize=(16,8),title='Open Price')

amazon['Open'].plot(label='Amazon')

google['Open'].plot(label='Google')

plt.legend();
apple['Volume'].plot(label='Apple',figsize=(16,8),title='Volume Traded')

amazon['Volume'].plot(label='Amazon')

google['Volume'].plot(label='Google')

plt.legend();
apple['Volume'].max()
apple['Volume'].argmax()
apple['Total Traded'] = apple['Open']*apple['Volume']

amazon['Total Traded'] = amazon['Open']*amazon['Volume']

google['Total Traded'] = google['Open']*google['Volume']
apple['Total Traded'].plot(label='Apple',figsize=(16,8))

amazon['Total Traded'].plot(label='Amazon')

google['Total Traded'].plot(label='Google')

plt.legend()

plt.ylabel('Total Traded')
amazon['Total Traded'].argmax()
amazon['MA50'] = amazon['Open'].rolling(50).mean()

amazon['MA200'] = amazon['Open'].rolling(200).mean()

amazon[['Open','MA50','MA200']].plot(label='gm',figsize=(16,8));
from pandas.plotting import scatter_matrix
tech_comp = pd.concat([apple['Open'],amazon['Open'],google['Open']],axis=1)
tech_comp.columns = ['Apple Open','Amazon Open','Google Open']
scatter_matrix(tech_comp,figsize=(8,8),alpha=0.2,hist_kwds={'bins':50});
apple['returns'] = (apple['Close'] / apple['Close'].shift(1) ) - 1
apple.head()
amazon['returns'] = (amazon['Close'] / amazon['Close'].shift(1) ) - 1
amazon.head()
# Now repeat for the other dataframes

amazon['returns'] = amazon['Close'].pct_change(1)

google['returns'] = google['Close'].pct_change(1)
google.head()
amazon = amazon.drop(['MA50', 'MA200'], axis=1)

amazon.head()
apple['returns'].hist(bins=50);
amazon['returns'].hist(bins=50);
google['returns'].hist(bins=50);
apple['returns'].hist(bins=100,label='Apple',figsize=(10,8),alpha=0.5)

amazon['returns'].hist(bins=100,label='Amazon',alpha=0.5)

google['returns'].hist(bins=100,label='Google',alpha=0.5)

plt.legend();
apple['returns'].plot(kind='kde',label='Apple',figsize=(12,6))

amazon['returns'].plot(kind='kde',label='Amazon')

google['returns'].plot(kind='kde',label='Google')

plt.legend();
box_df = pd.concat([apple['returns'],amazon['returns'],google['returns']],axis=1)

box_df.columns = ['Apple Returns','Amazon Returns','Google Returns']

box_df.plot(kind='box',figsize=(8,11),colormap='jet');
scatter_matrix(box_df,figsize=(8,8),alpha=0.2,hist_kwds={'bins':50});
box_df.plot(kind='scatter',x='Amazon Returns',y='Google Returns',alpha=0.4,figsize=(10,8));
box_df.plot(kind='scatter',x='Apple Returns',y='Google Returns',alpha=0.4,figsize=(10,8));
apple['Cumulative Return'] = (1 + apple['returns']).cumprod()
apple.head()
amazon['Cumulative Return'] = (1 + amazon['returns']).cumprod()

amazon.head()
google['Cumulative Return'] = (1 + google['returns']).cumprod()

google.head()
apple['Cumulative Return'].plot(label='Apple',figsize=(16,8),title='Cumulative Return')

amazon['Cumulative Return'].plot(label='Amazon')

google['Cumulative Return'].plot(label='Google')

plt.legend();
amazon['Cumulative Return'].plot(label='Amazon')

apple['Cumulative Return'].plot(label='Apple')

plt.legend();