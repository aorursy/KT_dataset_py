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
import datetime
%matplotlib inline

#Bank of America data frame
BAC = pd.read_csv('../input/BAC.csv')
#Citi Group data frame
C = pd.read_csv('../input/C.csv')
#Goldman Sach's data frame
GS = pd.read_csv('../input/GS.csv')
#JP Morgan Chase data frame
JPM = pd.read_csv('../input/JPM.csv')
#Morgan Stanley data frame
MS = pd.read_csv('../input/MS.csv')
#Wells Fargo data frame
WFC = pd.read_csv('../input/WFC.csv')

BAC.set_index('Date',inplace=True)
C.set_index('Date',inplace=True)
GS.set_index('Date',inplace=True)
JPM.set_index('Date',inplace=True)
MS.set_index('Date',inplace=True)
WFC.set_index('Date',inplace=True)

tickers = ['BAC','C','GS','JPM','MS','WFC']
bank_stocks = pd.concat([BAC,C,GS,JPM,MS,WFC],axis=1,keys=tickers)
bank_stocks.columns.names = ['Bank Ticker', 'Stock Info']

#Exploratory Data Analysis
#Maximun close price for each bank's stock throughout the time period
#for tick in tickers: print(tick,bank_stocks[tick]['Close'].max())
# (OR)
bank_stocks.xs(key='Close',axis=1,level='Stock Info').max()

#Creating 'Returns'Data Frame which contains the returns for each bank's stock (returns=price at a date/ price at a previous date)
returns = pd.DataFrame()
for tick in tickers:
    returns[tick+' Return'] = bank_stocks[tick]['Close'].pct_change()

returns.head()
#Pairplot on the returns DataFrame. Looks like Citi Group stocks stand out when compared to rest of the banks as it has no growth in its returns throughout the time
import seaborn as sns
#sns.pairplot(returns[1:])
# Dates on which each bank stock had their best and worst single day returns
returns.idxmin()
returns.idxmax()
#Highest Risk over the entire time period and during the year 2015
#High Std Deviation, High Risk
returns.std()
#During 2015
returns.loc['2015-01-01':'2015-12-31'].std()
#During 2006 to 2015, Citi Group has the higest risk
returns.loc['2006-01-01':'2015-12-31'].std()
#Distplot of 2015 returns for Morgan Stanley
#sns.distplot(returns.loc['2015-01-01':'2015-12-31']['MS Return'],color='green',bins=100)
#Distplot of 2008 returns for CitiGroup
#sns.distplot(returns.loc['2008-01-01':'2008-12-31']['C Return'],color='red',bins=100)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline

import plotly
import cufflinks as cf
cf.go_offline()

#Lineplot to see closing price for each bank from 2006 to 2015
for tick in tickers:
    bank_stocks[tick]['Close'].plot(figsize=(12,4),label=tick)

plt.legend()
# (OR) bank_stocks.xs(key='Close',axis=1,level='Stock Info').plot()
# plotly
bank_stocks.xs(key='Open',axis=1,level='Stock Info').iplot()
bank_stocks.xs(key='Close',axis=1,level='Stock Info').iplot()
bank_stocks.xs(key='Volume',axis=1,level='Stock Info').iplot()

#Rolling 30 day average against the Close Price for Bank Of America's stock for the year 2006
plt.figure(figsize=(12,6))
BAC['Close'].loc['2006-01-01':'2007-01-01'].rolling(window=30).mean().plot(label='30 Day Avg')
BAC['Close'].loc['2006-01-01':'2007-01-01'].plot(label='BAC CLOSE 2006-2007')
plt.legend()
#Rolling 30 day average against the Close Price for Bank Of America's stock for the year 2015
plt.figure(figsize=(12,6))
BAC['Close'].loc['2014-01-01':'2015-01-01'].rolling(window=30).mean().plot(label='30 Day Avg')
BAC['Close'].loc['2014-01-01':'2015-01-01'].plot(label='BAC CLOSE 2014-2015')
plt.legend()

#Heatmap of the correlation between the stocks Open Price
sns.heatmap(bank_stocks.xs(key='Open',axis=1,level='Stock Info').corr(),annot=True)
#Clustermap of the correlations between the stocks Open Price
sns.clustermap(bank_stocks.xs(key='Open',axis=1,level='Stock Info').corr(),annot=True)

#Heatmap of the correlation between the stocks Close Price
sns.heatmap(bank_stocks.xs(key='Close',axis=1,level='Stock Info').corr(),annot=True)
#Clustermap of the correlations between the stocks Close Price
sns.clustermap(bank_stocks.xs(key='Close',axis=1,level='Stock Info').corr(),annot=True)

#Heatmap of the correlation between the stocks Volume Price
sns.heatmap(bank_stocks.xs(key='Volume',axis=1,level='Stock Info').corr(),annot=True)
#Clustermap of the correlations between the stocks Volume Price
sns.clustermap(bank_stocks.xs(key='Volume',axis=1,level='Stock Info').corr(),annot=True)

close_corr = bank_stocks.xs(key='Close',axis=1,level='Stock Info').corr()
close_corr.iplot(kind='heatmap',colorscale='rdylbu')

#Technical Analysis Reports using cufflinks library
#Candle plot of each bank stock from Jan 1st 2015 to Jan 1st 2016
BAC[['Open','High','Low','Close']].loc['2015-01-01':'2016-01-01'].iplot(kind='candle',title='Candle plot for BAC')
C[['Open','High','Low','Close']].loc['2015-01-01':'2016-01-01'].iplot(kind='candle',title='Candle plot for C')
GS[['Open','High','Low','Close']].loc['2015-01-01':'2016-01-01'].iplot(kind='candle',title='Candle plot for GS')
JPM[['Open','High','Low','Close']].loc['2015-01-01':'2016-01-01'].iplot(kind='candle',title='Candle plot for JPM')
MS[['Open','High','Low','Close']].loc['2015-01-01':'2016-01-01'].iplot(kind='candle',title='Candle plot for MS')
WFC[['Open','High','Low','Close']].loc['2015-01-01':'2016-01-01'].iplot(kind='candle',title='Candle plot for WFC')

#Simple Moving Averages plot of each bank for the year 2015
BAC['Close'].loc['2015-01-01':'2016-01-01'].ta_plot(study='sma',periods=[13,21,55],title='Simple Moving Averages BAC')
C['Close'].loc['2015-01-01':'2016-01-01'].ta_plot(study='sma',periods=[13,21,55],title='Simple Moving Averages C')
GS['Close'].loc['2015-01-01':'2016-01-01'].ta_plot(study='sma',periods=[13,21,55],title='Simple Moving Averages GS')
JPM['Close'].loc['2015-01-01':'2016-01-01'].ta_plot(study='sma',periods=[13,21,55],title='Simple Moving Averages JPM')
MS['Close'].loc['2015-01-01':'2016-01-01'].ta_plot(study='sma',periods=[13,21,55],title='Simple Moving Averages MS')
WFC['Close'].loc['2015-01-01':'2016-01-01'].ta_plot(study='sma',periods=[13,21,55],title='Simple Moving Averages WFC')

#Bollinger Band Plot for each bank for the year 2015
BAC['Close'].loc['2015-01-01':'2016-01-01'].ta_plot(study='boll',title='BAC')
C['Close'].loc['2015-01-01':'2016-01-01'].ta_plot(study='boll',title='C')
GS['Close'].loc['2015-01-01':'2016-01-01'].ta_plot(study='boll',title='GS')
JPM['Close'].loc['2015-01-01':'2016-01-01'].ta_plot(study='boll',title='JPM')
MS['Close'].loc['2015-01-01':'2016-01-01'].ta_plot(study='boll',title='MS')
WFC['Close'].loc['2015-01-01':'2016-01-01'].ta_plot(study='boll',title='WFC')

