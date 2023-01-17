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
from pandas_datareader import data, wb
import pandas as pd
import numpy as np
import datetime
%matplotlib inline
start = datetime.datetime(2006,1,1)
end = datetime.datetime.now()
'''
# Bank of America
BAC = data.DataReader("BAC", 'morningstar', start, end)

# CitiGroup
C = data.DataReader("C", 'morningstar', start, end)

# Goldman Sachs
GS = data.DataReader("GS", 'morningstar', start, end)

# JPMorgan Chase
JPM = data.DataReader("JPM", 'morningstar', start, end)

# Morgan Stanley
MS = data.DataReader("MS", 'morningstar', start, end)

# Wells Fargo
WFC = data.DataReader("WFC", 'morningstar', start, end)

'''
df = pd.read_pickle('../input/all_banks.pickle')
df.head()
df.xs(key='Close',axis=1,level='Stock Info').max()
returns = pd.DataFrame()
tickers = ['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC']
for tick in tickers:
    returns[tick+' Return'] = df[tick]['Close'].pct_change()
returns.head()
import seaborn as sns
sns.pairplot(returns[1:])
# Worst Drop (4 of them on Inauguration day)
returns.idxmin()
# Best Single Day Gain
# citigroup stock split in May 2011, but also JPM day after inauguration.
returns.idxmax()
returns.std()
sns.distplot(returns.loc['2017-05-01':'2018-04-30']['MS Return'],color='green',bins=100)
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
    df[tick]['Close'].plot(figsize=(12,4),label=tick)
plt.legend()
df.xs(key='Close',axis=1,level='Stock Info').plot(figsize=(12,4))
plt.figure(figsize=(12,6))
df['BAC']['Close'].loc['2008-01-01':'2009-01-01'].rolling(window=30).mean().plot(label='30 Day Avg')
df['BAC']['Close'].loc['2008-01-01':'2009-01-01'].plot(label='BAC CLOSE')
plt.legend()
plt.figure(figsize=(12,6))
df['C']['Close'].loc['2008-01-01':'2009-01-01'].rolling(window=30).mean().plot(label='30 Day Avg')
df['C']['Close'].loc['2008-01-01':'2009-01-01'].plot(label='C CLOSE')
plt.legend()
# plotly
df.xs(key='Close',axis=1,level='Stock Info').iplot()
sns.heatmap(df.xs(key='Close',axis=1,level='Stock Info').corr(),annot=True)
sns.clustermap(df.xs(key='Close',axis=1,level='Stock Info').corr(),annot=True)
close_corr = df.xs(key='Close',axis=1,level='Stock Info').corr()
close_corr.iplot(kind='heatmap',colorscale='rdylbu')
df['BAC'][['Open', 'High', 'Low', 'Close']].loc['2017-01-01':'2018-04-30'].iplot(kind='candle')
df['MS']['Close'].loc['2017-01-01':'2018-01-01'].ta_plot(study='sma',periods=[13,21,55],title='Simple Moving Averages')
df['C']['Close'].loc['2017-01-01':'2018-01-01'].ta_plot(study='boll')
