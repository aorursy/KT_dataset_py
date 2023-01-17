# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set_style('whitegrid')

color = sns.color_palette()

import plotly

import plotly.graph_objs as go

import cufflinks as cf

cf.go_offline()



import os

from configparser import ConfigParser

import requests

from datetime import datetime



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Dataframe having list of stocks with their ticker symbols

stocklist = pd.DataFrame([['Bank of America', 'BAC'], ['CitiGroup','C'], ['Goldman Sachs', 'GS'], ['JPMorgan Chase', 'JPM'],

                                  ['Morgan Stanley','MS'], ['Wells Fargo','WFC']], columns= ['stock','symbol'])

stocklist
# Reading the stock price CSV files

for i,val in stocklist.iterrows():

    #print(val['symbol'])

    path = '../input/' +  val['symbol'] + '.csv'

    vars()[val['symbol']] = pd.read_csv(path).rename(columns= {'Unnamed: 0':'date'}).set_index('date')
# Concatenating the files together in a single dataframe

tickers = stocklist['symbol'].tolist()

bank_stocks = pd.concat([BAC, C, GS, JPM, MS, WFC],axis=1,keys=tickers)

bank_stocks.columns.names = ['Stock','Info']

bank_stocks.head()
# Making a plotly plot of the Close Price vs Time for the years 2007-2011 showing the financial crisis in 2008.

layout = go.Layout(

    title=go.layout.Title(

        text='Close Price (2007-2011) vs Time',

    ))

bank_stocks.xs(key='close',axis=1,level='Info').loc['2007-01-01':'2012-01-01'].iplot(layout=layout)
# Close Price and Smoothed Close Price of BAC during the financial crisis

plt.figure(figsize=(12,6))

plt.title('Downtrend of BAC stock during the financial crisis', size=14)

BAC['close'].loc['2008-01-01':'2009-01-01'].rolling(window=30).mean().plot(label='30 Day MA')

BAC['close'].loc['2008-01-01':'2009-01-01'].plot(label='BAC CLOSE')

plt.legend()

plt.show()
# Finding max and min Close price of each stock and plotting it

maxs = bank_stocks.xs(key='close',axis=1,level='Info').max().reset_index()

maxs.columns= ['stock', 'max']

mins = bank_stocks.xs(key='close',axis=1,level='Info').min().reset_index()

mins.columns= ['stock', 'min']
m = maxs.merge(mins, on='stock').set_index('stock')

m.plot(kind='line', figsize=(6,5), style='o')

plt.xlabel('Stocks')

plt.ylabel('Close Price')

plt.title('Max and Min Close Price')

# Loc, Value to xticks

plt.xticks(range(m.shape[0]), m.index)

plt.xlim([-1, len(m)])

plt.show()
print(m)
# Boxplots to see a comparison of the ranges and quartiles of stock prices

b =bank_stocks.xs(key='close',axis=1,level='Info')

c = b.unstack().reset_index()

c = c.rename(columns={0:'val'})

plt.figure(figsize=(12,6))

sns.boxplot(x='Stock', y="val", data=c, color='crimson')

plt.ylabel('Close Price', fontsize=12)

plt.xlabel('Stock', fontsize=12)

plt.title("Box Plots of Close Prices", fontsize=15)

plt.show()
# Violinplots to get idea of the distribution of stock prices

plt.figure(figsize=(12,6))

sns.violinplot(x='Stock', y="val", data=c, color='gold', alpha=0.9)

plt.ylabel('Close Price', fontsize=12)

plt.xlabel('Stocks', fontsize=12)

plt.title("Violin Plots of Close Prices", fontsize=15)

plt.show()
# Finding the daily returns of each stock from the close prices

returns = pd.DataFrame()

for tick in tickers:

    returns[tick+' Return'] = bank_stocks[tick]['close'].pct_change()

    returns = returns.dropna(axis=0)

returns.head()
# Pair-wise scatter plot of returns of each stock

sns.pairplot(returns)

plt.show()
# Candlestick chart of Citibank stock in Nov 2008.

C[['open', 'high', 'low', 'close']].loc['2008-10-01':'2009-01-01'].iplot(kind='candle')
# Worst Drop in returns



ret_idx_min = returns.idxmin().reset_index()

ret_idx_min.columns = ['stock', 'date']

ret_min = returns.min().reset_index()

ret_min.columns = ['stock', 'min_return']

ret_min['min_return'] = ret_min['min_return'].apply(lambda x: '%.2f'%x)

ret_idx_min.merge(ret_min, on='stock').set_index('stock')
# Best single day gains



ret_idx_max = returns.idxmax().reset_index()

ret_idx_max.columns = ['stock', 'date']

ret_max = returns.max().reset_index()

ret_max.columns = ['stock', 'max_return']

ret_max['max_return'] = ret_max['max_return'].apply(lambda x: '%.2f'%x)

ret_idx_max.merge(ret_max, on='stock').set_index('stock')
# Some idea of sudden Recuperation-- Difference between max and min gain days

pd.to_datetime(returns.idxmax()) - pd.to_datetime(returns.idxmin())
# Distribution plots of stock returns during crisis and stable periods

fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(16,6))

plt.suptitle('Comparison of returns during stable (left) and crisis period (right)', size=15)

sns.distplot(returns.loc['2018-01-01':'2018-12-31']['MS Return'],color='brown',bins=100, ax= ax[0])

sns.distplot(returns.loc['2008-01-01':'2008-12-31']['C Return'],color='red',bins=100, ax=ax[1])

plt.show()
# Standard Deviation of returns as a measure of riskness of the stocks

returns.std().map(lambda x :'%.2f'%x)
# Bollinger Bands (for risk assessement) of Citibank stock

C['close'].loc['2007-01-01':'2009-12-31'].ta_plot(study='boll')
# Standard Deviation during Jan 2018- Dec 2018

returns.loc['2018-01-01':'2018-12-31'].std().map(lambda x :'%.3f'%x)
# Heatmap of correlation between stock prices

plt.figure(figsize=(7,5))

sns.heatmap(bank_stocks.xs(key='close',axis=1,level='Info').corr(),annot=True, cmap='YlGnBu')

plt.show()
# Dendogram of correlation between stock prices

sns.clustermap(bank_stocks.xs(key='close',axis=1,level='Info').corr(),annot=True, figsize=(7,7))