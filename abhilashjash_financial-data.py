# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas_datareader import data, wb

import datetime

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
start = datetime.datetime(2006, 1, 1)

end = datetime.datetime(2016, 1, 1)
# Bank of America

BAC = data.DataReader("BAC", 'yahoo', start, end)



# CitiGroup

C = data.DataReader("C", 'yahoo', start, end)



# Goldman Sachs

GS = data.DataReader("GS", 'yahoo', start, end)



# JPMorgan Chase

JPM = data.DataReader("JPM", 'yahoo', start, end)



# Morgan Stanley

MS = data.DataReader("MS", 'yahoo', start, end)



# Wells Fargo

WFC = data.DataReader("WFC", 'yahoo', start, end)
tickers = ['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC']
bank_stocks = pd.concat([BAC, C, GS, JPM, MS, WFC],axis=1,keys=tickers)
bank_stocks.columns.names = ['Bank Ticker','Stock Info']
bank_stocks.head()
bank_stocks.xs(key='Close',axis=1,level='Stock Info').max()
returns = pd.DataFrame()
for tick in tickers:

    returns[tick+' Return'] = bank_stocks[tick]['Close'].pct_change()

returns.head()
#returns[1:]

import seaborn as sns

sns.pairplot(returns[1:])
# Worst Drop (4 of them on Inauguration day)

returns.idxmin()
# Best Single Day Gain

# citigroup stock split in May 2011, but also JPM day after inauguration.

returns.idxmax()
returns.std() # Citigroup riskiest
returns['2015-01-01':'2015-12-31'].std() # Very similar risk profiles, but Morgan Stanley or BofA
sns.distplot(returns['2015-01-01':'2015-12-31']['MS Return'],color='green',bins=100)
sns.distplot(returns['2008-01-01':'2008-12-31']['C Return'],color='red',bins=100)
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
bank_stocks.xs(key='Close',axis=1,level='Stock Info').plot(figsize = (20,10))
# plotly

bank_stocks.xs(key='Close',axis=1,level='Stock Info').iplot()