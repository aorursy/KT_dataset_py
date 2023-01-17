# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def plot_multiple_stocks(symbols, date_from, date_to):
    
    dates = pd.date_range(date_from, date_to, freq = 'B')
    dates_df = pd.DataFrame({'Date': dates})
    
    for symbol in symbols:
        data = pd.read_csv('/Users/subleenkaur/Downloads/Data/Stocks/{}.us.txt'.format(symbol), na_values=['nan'])
        data = data[['Date', 'Close']]
        data = data.rename(columns = {'Close': symbol})
        data['Date'] = pd.to_datetime(data['Date'])
        dates_df = pd.merge(data, dates_df, on = 'Date', how = 'right' )
        
    return dates_df

symbols = ['aapl', 'goog', 'ibm']
date_from = '2016-01-02'
date_to = '2016-12-31'

df = plot_multiple_stocks(symbols , date_from, date_to)
df.fillna(method = 'pad')
df.set_index('Date').plot()

plt.show()
def plot_single_stock(symbol):
    data = pd.read_csv('/Users/subleenkaur/Downloads/Data/Stocks/{}.us.txt'.format(symbol))
    data[['Close']].plot()
    plt.show()
plot_single_stock('aapl')
# Normalized Stocks - base value from 2016-01-04
df.set_index('Date' , inplace = True)
print(df.iloc[1,:])
df = df / df.iloc[1 , :]
df.interpolate().plot()
plt.show()
def daily_returns(df):
    dr = df.copy()
    df = dr[:-1].values / dr[1:] - 1
    return dr
symbols = ['aapl']
date_from = '2016-01-02'
date_to = '2016-12-31'

df = plot_multiple_stocks(symbols , date_from, date_to)
df.set_index('Date' , inplace = True)
dr = daily_returns(df)

dr = dr.interpolate()

dr.interpolate().plot()
plt.title('Apple Daily Returns')
plt.show()
dr.hist(bins = 20)
plt.show()
def cum_returns(df):
    dr = df.copy()
    df.cumsum()
    return dr
symbols = ['aapl']
date_from = '2016-01-02'
date_to = '2016-12-31'

df = plot_multiple_stocks(symbols , date_from, date_to)
df.set_index('Date' , inplace = True)
dr = cum_returns(df)

dr.plot()
plt.title('Apple Cumulative Returns')
plt.show()
#Scatterplots
symbols = ['aapl', 'goog']
date_from = '2016-01-02'
date_to = '2016-12-31'

df = plot_multiple_stocks(symbols , date_from, date_to)
df.set_index('Date' , inplace = True)
dr = daily_returns(df)
dr.plot(kind='scatter',x='goog', y='aapl')
plt.show()
#Calculating Simple Moving Averages
def get_SMA(df, n_days):
    dr = df.copy()
    dr.rolling(n_days).mean()
    return dr


symbols = ['aapl']
date_from = '2016-01-02'
date_to = '2016-12-31'

df = plot_multiple_stocks(symbols , date_from, date_to)
df.set_index('Date' , inplace = True)
dr = get_SME(df, 10)

dr.plot()
plt.title("Simple Moving Averages")
plt.show()
def get_EMA(df, n_days):
    dm = df.ewm( span = n_days, min_periods = n_days - 1).mean()
    return dm
symbols = ['aapl']
date_from = '2016-01-02'
date_to = '2016-12-31'

df = plot_multiple_stocks(symbols , date_from, date_to)
df.set_index('Date' , inplace = True)
dr = get_EMA(df, 10)

print(dr.tail())
dr.plot()
plt.title("Exponential Moving Averages")
plt.show()
def get_ROC(df, n_days):
    dn = df.diff(n_days)
    dd = df.shift(n_days)
    return dn/dd
symbols = ['aapl']
date_from = '2016-01-02'
date_to = '2016-12-31'

df = plot_multiple_stocks(symbols , date_from, date_to)
df.set_index('Date' , inplace = True)
dr = get_ROC(df, 10)
print(dr)

dr.plot()
plt.title("Rate of Change")
plt.show()
