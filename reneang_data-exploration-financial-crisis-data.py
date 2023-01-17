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



#Other libraries 



import pandas as pd

import numpy as np

import seaborn as sns

import datetime

%matplotlib inline

from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

print(__version__) # requires version >= 1.9.0

import cufflinks as cf

init_notebook_mode(connected=True)

cf.go_offline()
bank_stocks = pd.read_pickle('../input/all_banks') 

bank_stocks.head()

bank_stocks.xs(key='Close',axis=1,level='Stock Info').head(3)
bank_stocks.xs(key='Close',axis=1,level='Stock Info').max()
tickers = ['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC']

returns = pd.DataFrame()



for x in tickers:

    returns[x + ' Return'] = bank_stocks.xs('Close',axis=1,level=1)[x].pct_change()

returns.head()
sns.pairplot(returns[1:])
# Worst Drop of returns (4 of them on Inauguration day)

returns.idxmin()
# Best Drop of returns (4 of them on Inauguration day)

returns.idxmax()
# Citigroup riskiest

returns.std()
returns.loc['2015-01-01':'2015-12-31'].std() 
returns.loc['2015-01-01':'2015-12-31'].std() # Very similar risk profiles, but Morgan Stanley or BofA
sns.distplot(returns.loc['2015-01-01':'2015-12-31']['MS Return'],color='green',bins=100)
sns.distplot(returns.loc['2008-01-01':'2008-12-31']['C Return'],color='red',bins=100)
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot

init_notebook_mode(connected=True)

bank_stocks.xs('Close',axis=1,level=1).iplot()
# This same plot can be done with a for loop

import matplotlib.pyplot as plt

for tick in tickers:

     bank_stocks[tick]['Close'].plot(label=tick,figsize=(6,2))

plt.legend()


rolling_avg = pd.DataFrame()

rolling_avg['30 Day Avg'] = bank_stocks.loc['2008-01-01':'2008-12-31'].xs('Close',axis=1,level=1)['BAC'].rolling(window=30).mean()



rolling_avg['30 Day Avg'].plot(figsize=(12,6),label='30 Day Avg')

bank_stocks.loc['2008-01-01':'2008-12-31'].xs('Close',axis=1,level=1)['BAC'].plot(figsize=(12,6),label='BofA Close')

plt.legend()

sns.heatmap(bank_stocks.xs('Close',axis=1,level=1).corr(),annot=True)
sns.clustermap(bank_stocks.xs('Close',axis=1,level=1).corr(),annot=True)