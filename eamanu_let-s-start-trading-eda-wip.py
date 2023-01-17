# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn  as sns
import gc
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
(market_train_df, news_train_df) = env.get_training_data()
# Let's do a copy because the above methods could be call once
market_train = market_train_df
news_train = news_train_df
print("shape market_train ", market_train.shape)
market_train.head(5)
print(market_train.info())
# Let's see the the NaN values. 
print(market_train.isnull().sum())
print("The NaN values on returnsClosePrevMktres1 represent the: %f" % (15980/4072956))
print("The NaN values on returnsClosePrevMktres10 represent the: %f" % (93010/4072956))
market_train.dtypes
market_train.nunique()
market_train.describe(include='all')
aapl_jan = market_train.query("time.dt.year == 2010 and assetCode == 'AAPL.O'")
aapl_jan
plt.figure(figsize=(10,6))
# plt.plot(range(len(aapl_jan.time)), aapl_jan.close, label='Close price')
# plt.plot(range(len(aapl_jan.time)), aapl_jan.open, label='Open price')
plt.title("Opening and closing price")
plt.plot(aapl_jan.time, aapl_jan.open, label='Open price')
plt.plot(aapl_jan.time, aapl_jan.close, label='Close price')
plt.legend()
plt.show()
plt.figure(figsize=(10,6))
plt.title("Opening and closing return mtres 1")
plt.bar(range(len(aapl_jan.time)), aapl_jan.returnsOpenPrevMktres1, label='Return Open price')
plt.bar(range(len(aapl_jan.time)), aapl_jan.returnsClosePrevMktres1, label='Return Close price')
plt.legend()
plt.show()
aapl_daily_pct_change = aapl_jan.close / aapl_jan.close.shift(1) - 1
aapl_daily_pct_change.hist(bins=50)
market_train = market_train.assign(
    daily_percent_price=market_train.groupby('assetCode',
                                            as_index=False).apply(lambda x: x.close / x.close.shift(1) - 1)
    .reset_index(0, drop=True)
)
plt.figure(figsize=(12,8))
ax1 = plt.subplot(221)
market_train.query("time.dt.year == 2016 and assetCode == 'AAPL.O'")['daily_percent_price'].hist(bins=50)
ax2 = plt.subplot(222)
market_train.query("time.dt.year == 2016 and assetCode == 'YPF.N'")['daily_percent_price'].hist(bins=50)
ax3 = plt.subplot(223)
market_train.query("time.dt.year == 2016 and assetCode == 'A.N'")['daily_percent_price'].hist(bins=50)
ax4 = plt.subplot(224)
market_train.query("time.dt.year == 2016 and assetCode == 'CMC.N'")['daily_percent_price'].hist(bins=50)
plt.show()
market_train = market_train.assign(
    sma_5=market_train.groupby(['assetCode'], 
                     as_index=False)[['close']]
    .rolling(window=5).mean().reset_index(0, drop=True))
market_train = market_train.assign(
    ema_10=market_train.groupby(['assetCode'], as_index=False)
    .apply(lambda g: g.close.ewm(10).mean()).reset_index(0, drop=True)
)
market_train = market_train.assign(
    ema_20=market_train.groupby(['assetCode'], as_index=False)
    .apply(lambda g: g.close.ewm(20).mean()).reset_index(0, drop=True)
)
market_train = market_train.assign(
    ema_30=market_train.groupby(['assetCode'], as_index=False)
    .apply(lambda g: g.close.ewm(30).mean()).reset_index(0, drop=True)
)
market_train = market_train.assign(
    ema_50=market_train.groupby(['assetCode'], as_index=False)
    .apply(lambda g: g.close.ewm(50).mean()).reset_index(0, drop=True)
)
market_train = market_train.assign(
    ema_100=market_train.groupby(['assetCode'], as_index=False)
    .apply(lambda g: g.close.ewm(100).mean()).reset_index(0, drop=True)
)
market_train = market_train.assign(
    ema_200=market_train.groupby(['assetCode'], as_index=False)
    .apply(lambda g: g.close.ewm(200).mean()).reset_index(0, drop=True)
)
plt.figure(figsize=(10, 8))
plt.title("Moving average for AAPL. 2016")
market_train.query("time.dt.year == 2016 and assetCode == 'AAPL.O'").close.plot(legend=True)
market_train.query("time.dt.year == 2016 and assetCode == 'AAPL.O'").sma_5.plot(legend=True)
market_train.query("time.dt.year == 2016 and assetCode == 'AAPL.O'").ema_10.plot(legend=True)
market_train.query("time.dt.year == 2016 and assetCode == 'AAPL.O'").ema_20.plot(legend=True)
market_train.query("time.dt.year == 2016 and assetCode == 'AAPL.O'").ema_30.plot(legend=True)
market_train.query("time.dt.year == 2016 and assetCode == 'AAPL.O'").ema_50.plot(legend=True)
market_train.query("time.dt.year == 2016 and assetCode == 'AAPL.O'").ema_100.plot(legend=True)
market_train.query("time.dt.year == 2016 and assetCode == 'AAPL.O'").ema_200.plot(legend=True)
plt.show()
market_train = market_train.assign(
    ema_26=market_train.groupby(['assetCode'], as_index=False)
    .apply(lambda g: g.close.ewm(26).mean()).reset_index(0, drop=True)
)
market_train = market_train.assign(
    ema_12=market_train.groupby(['assetCode'], as_index=False)
    .apply(lambda g: g.close.ewm(12).mean()).reset_index(0, drop=True)
)
market_train['MACD'] = market_train.ema_12 - market_train.ema_26
market_train.tail(1)
market_train = market_train.assign(
    signal_line_macd=market_train.groupby(['assetCode'], as_index=False)
    .apply(lambda g: g.MACD.ewm(9).mean()).reset_index(0, drop=True)
)
query = market_train.query("time.dt.year == 2011 and assetCode == 'AAPL.O'")
f1, ax1 = plt.subplots(figsize=(8,4))
ax1.plot(query.index, query.close, color='black', lw=2, label='Close Price')
ax1.legend(loc='upper right')
ax1.set(title="Close Price for AAPL. 2011", ylabel='Price')
f2, ax2 = plt.subplots(figsize=(8,4))
ax2.plot(query.index, query.MACD, color='green', lw=1, label='MACD Line (26, 12)')
ax2.plot(query.index, query.signal_line_macd, color='purple', lw=1, label='Signal')
ax2.fill_between(query.index, query.MACD - query.signal_line_macd, color='gray', alpha=0.5, label='MACD Histogram')
ax2.set(title='MACD for AAPL. 2011', ylabel='MACD')
ax2.legend(loc='upper right')
plt.show()
market_train['signal_crossover_macd'] = 0.0
market_train.signal_crossover_macd = np.where(market_train.MACD > market_train.signal_line_macd, 1.0, 0.0)
market_train['signal_crossover_macd'] = market_train.groupby(['assetCode'], as_index=False)['signal_crossover_macd'].diff().reset_index(0, drop=True)
def relative_strength_index(close, n):
    """Calculate Relative Strength Index(RSI) for given data.
    
    :param df: pandas.Series
    :param n: 
    :return: pandas.DataFrame
    """
    buf = pd.DataFrame()
    buf['close'] = close
    buf['diff'] = buf.close.diff()
    mask = buf['diff'] < 0
    buf['high'] = abs(buf['diff'].mask(mask))
    buf['low'] = abs(buf['diff'].mask(~mask))
    buf['high'] = buf['high'].fillna(0)
    buf['low'] = buf['low'].fillna(0)
    posrs = buf['high'].ewm(span=n, min_periods=n).mean()
    negrs = buf['low'].ewm(span=n, min_periods=n).mean()
    buf['rsi'] = posrs / (posrs + negrs)
    return buf.rsi

market_train = market_train.assign(
    rsi=market_train.groupby(['assetCode'], as_index=False)
    .apply(lambda g: relative_strength_index(g.close, 14)).reset_index(0, drop=True)
)
market_train.tail()
market_train['month'] = market_train['time'].apply(lambda x: x.month)
market_train.groupby('month').sum()['volume'].plot(figsize=(10,8))
market_train['year'] = market_train['time'].apply(lambda x: x.year)
# market_train.groupby(['year', 'month']).sum()['volume'].heatmap(figsize=(10,8))
df = market_train.pivot_table(index='year', columns='month', values='volume', aggfunc=np.sum)
plt.figure(figsize=(10, 8))
sns.heatmap(df, annot=False, fmt=".1f")
plt.show()
market_train['day'] = market_train['time'].apply(lambda x: x.dayofweek)
market_train.groupby('day').sum()['volume'].plot(figsize=(10,8))
market_train['ticket'] = market_train.assetCode.str.split('.', expand=True).iloc[:, 0]
market_train['market'] = market_train.assetCode.str.split('.', expand=True).iloc[:, 1]
market_train.market.value_counts()






