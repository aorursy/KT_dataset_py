import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.finance import candlestick2_ohlc

# from mpl_finance import candlestick_ohlc

import seaborn as sns

from datetime import datetime
coins = {'BTC':'bitcoin', 'ETH':'ethereum', 'XRP':'ripple', 'LTC':'litecoin', 'DASH':'dash'}

ticker = list(coins.keys())
raw = {t: pd.read_csv("../input/{}_price.csv".format(coins[ticker[index]].lower())) 

       for index, t in enumerate(ticker)}
for t in ticker:

    raw[t].index = pd.to_datetime(raw[t]['Date'], format='%b %d, %Y')

    raw[t].drop('Date', axis=1, inplace=True)

    raw[t]['Volume'] = pd.to_numeric(raw[t]['Volume'].str.replace(',','').str.replace('-',''))

    raw[t]['Market Cap'] = pd.to_numeric(raw[t]['Market Cap'].str.replace(',','').str.replace('-',''))
for t in ticker:

    print(t, raw[t].index[-1])
data = pd.concat(raw.values(), axis=1, keys=ticker)

data.columns.names = ['Ticker', 'Info']
data.tail().sort_index(ascending=False)
data.xs(key='Close', axis=1, level='Info').max()
returns = pd.DataFrame(index=data.index)
for t in ticker:

    returns[t+'_return'] = data[t]['Close'].pct_change()
print("NaN count: "+str(returns.loc['2016-10-01 00:00:00':'2017-09-30 00:00:00'].isnull().sum().sum()))
sns.set_style('whitegrid')

sns.set_context("talk")
sns.pairplot(returns.loc['2016-10-01 00:00:00':'2017-09-30 00:00:00'])
returns.idxmax()
returns.idxmin()
returns.loc['2016-10-01 00:00:00':'2017-09-30 00:00:00'].std()
fig, ax = plt.subplots(figsize=(14,6),sharey=True)

for t in ticker[:3]:

    ax = sns.distplot(returns[t+'_return'].loc['2016-10-01 00:00:00':'2017-09-30 00:00:00'], kde=True, label=t, bins=60)

ax.set_xlim([-0.2, 0.2])

ax.set_xlabel("Return")

ax.legend()

plt.title("Histogram")

plt.show()
def period(x):

    if x % 2 == 0:

        return x/2-1

    else:

        return (x-1)/2
returns['Period'] = pd.DatetimeIndex(returns.index).month

returns['Period'] = returns['Period'].apply(period)
fig, ax = plt.subplots(figsize=(8, 6))

sns.boxplot(x="Period", y="BTC_return", data=returns)

sns.swarmplot(x="Period", y="BTC_return", data=returns, size=2, color=".3", linewidth=0)

ax.set_ylim([-0.15, 0.15])

ax.xaxis.grid(True)
for t in ticker:

    data[t]['Close'].plot(label=t, figsize=(14,6))

plt.legend()
data_AugSep = data.loc['2017-08-01 00:00:00':'2017-09-30 00:00:00']
fig, ax = plt.subplots(figsize=(14,6))

candlestick2_ohlc(ax, data_AugSep['BTC']['Open'], data_AugSep['BTC']['High'], data_AugSep['BTC']['Low'], data_AugSep['BTC']['Close'], width=0.6, colorup='g', colordown='r')
data['BTC']['Close'].loc['2017-01-01 00:00:00':'2017-09-30 00:00:00'].rolling(window=13).mean().plot(label="13 Day", figsize=(14,6))

data['BTC']['Close'].loc['2017-01-01 00:00:00':'2017-09-30 00:00:00'].rolling(window=21).mean().plot(label="21 Day", figsize=(14,6))

data['BTC']['Close'].loc['2017-01-01 00:00:00':'2017-09-30 00:00:00'].rolling(window=55).mean().plot(label="55 Day", figsize=(14,6))

data['BTC']['Close'].loc['2017-01-01 00:00:00':'2017-09-30 00:00:00'].plot(label="Close")

plt.legend()
sns.heatmap(data.xs(key='Close', axis=1, level='Info').corr(), annot=True)
sns.clustermap(data.xs(key='Close', axis=1, level='Info').corr(), annot=True)
for t in ticker:

    data[t]['Volume'].loc['2017-01-01 00:00:00':'2017-09-30 00:00:00'].plot(label=t, figsize=(14,6))

plt.legend()
for t in ticker:

    data[t]['Market Cap'].loc['2017-01-01 00:00:00':'2017-09-30 00:00:00'].plot(label=t, figsize=(14,6))

plt.legend()