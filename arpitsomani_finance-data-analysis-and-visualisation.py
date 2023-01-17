from pandas_datareader import data, wb

import pandas as pd

import numpy as np

import datetime

%matplotlib inline
df = pd.read_pickle('../input/dataset/all_banks')

df['BAC']
# Bank of America

BAC = df['BAC']



# CitiGroup

C = df['C']



# Goldman Sachs

GS = df['GS']



# JPMorgan Chase

JPM = df['JPM']



# Morgan Stanley

MS = df['MS']



# Wells Fargo

WFC = df['WFC']
tickers = ['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC']
bank_stocks = pd.concat([BAC, C, GS, JPM, MS, WFC],axis=1,keys=tickers)
bank_stocks.columns.names = ['Bank Ticker','Stock Info']
bank_stocks.head()
bank_stocks.xs(key='Close',axis=1,level='Stock Info').max()
returns = pd.DataFrame()
for tick in tickers:

    returns[tick+' Return'] = bank_stocks[tick]['Close'].pct_change()

returns.head()
import seaborn as sns

sns.pairplot(returns[1:])
returns.idxmin()
returns.idxmax()
returns.std()
returns.loc['2015-01-01':'2015-12-31'].std()
sns.distplot(returns.loc['2015-01-01':'2015-12-31']['MS Return'],color='green',bins=100)
sns.distplot(returns.loc['2008-01-01':'2008-12-31']['C Return'],color='red',bins=100)
import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



# Optional Plotly Method Imports

import plotly

import cufflinks as cf

cf.go_offline()
for tick in tickers:

    bank_stocks[tick]['Close'].plot(figsize=(12,4),label=tick)

plt.legend()
bank_stocks.xs(key='Close',axis=1,level='Stock Info').plot()
bank_stocks.xs(key='Close',axis=1,level='Stock Info').iplot()
plt.figure(figsize=(12,6))

BAC['Close'].loc['2008-01-01':'2009-01-01'].rolling(window=30).mean().plot(label='30 Day Avg')

BAC['Close'].loc['2008-01-01':'2009-01-01'].plot(label='BAC CLOSE')

plt.legend()
sns.heatmap(bank_stocks.xs(key='Close',axis=1,level='Stock Info').corr(),annot=True)
sns.clustermap(bank_stocks.xs(key='Close',axis=1,level='Stock Info').corr(),annot=True)
close_corr = bank_stocks.xs(key='Close',axis=1,level='Stock Info').corr()

close_corr.iplot(kind='heatmap',colorscale='rdylbu')
BAC[['Open', 'High', 'Low', 'Close']].loc['2015-01-01':'2016-01-01'].iplot(kind='candle')
MS['Close'].loc['2015-01-01':'2016-01-01'].ta_plot(study='sma',periods=[13,21,55],title='Simple Moving Averages')
BAC['Close'].loc['2015-01-01':'2016-01-01'].ta_plot(study='boll')