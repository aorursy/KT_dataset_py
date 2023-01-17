from pandas_datareader import data, wb

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import datetime



%matplotlib inline
start = datetime.datetime(2006, 1, 1)

end = datetime.datetime(2016, 1, 1)
# Bank of America

BAC = data.DataReader("BAC", 'stooq', start, end)



# CitiGroup

C = data.DataReader("C", 'stooq', start, end)



# Goldman Sachs

GS = data.DataReader("GS", 'stooq', start, end)



# JPMorgan Chase

JPM = data.DataReader("JPM", 'stooq', start, end)



# Morgan Stanley

MS = data.DataReader("MS", 'stooq', start, end)



# Wells Fargo

WFC = data.DataReader("WFC", 'stooq', start, end)
df = data.DataReader(['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC'],'stooq', start, end)
tickers = ['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC']

bank_stocks = pd.concat([BAC, C, GS, JPM, MS, WFC],axis=1,keys=tickers)

bank_stocks.columns.names = ['Bank Ticker','Stock Info']

bank_stocks.head()
returns = pd.DataFrame()

for tick in tickers:

    returns[tick+' Return'] = bank_stocks[tick]['Close'].pct_change()

returns.head()
sns.pairplot(returns[1:])

plt.show()
sns.heatmap(bank_stocks.xs(key='Close',axis=1,level='Stock Info').corr(),annot=True)

plt.show()
plt.figure(figsize=(20,8))

for tick in tickers:

    bank_stocks[tick]['Close'].plot(label=tick)

plt.legend()

plt.show()
BAC.head()
plt.figure(figsize=(18,8))

BAC['Close'].loc['2010-01-01':'2008-01-01'].rolling(window=30,center=True).mean().plot(label='30 Day Avg')

BAC['Close'].loc['2010-01-01':'2008-01-01'].plot(label='BAC CLOSE')

plt.legend()

plt.show()