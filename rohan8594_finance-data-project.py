import pandas as pd

import numpy as np

import datetime

%matplotlib inline
bank_stocks = pd.read_pickle('../input/all_banks')
tickers = ['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC']
bank_stocks.head()
bank_stocks.xs('Close', level=1, axis=1).max()
returns = pd.DataFrame()
for x in tickers:

    returns[x + ' Return'] = bank_stocks.xs('Close',axis=1,level=1)[x].pct_change()

returns.head()
import seaborn as sns

sns.set_style('whitegrid')
sns.pairplot(returns[1:])
returns.idxmin()
returns.idxmax()
returns.std() 
returns.loc['2015-01-01':'2015-12-31'].std()
sns.distplot(returns.loc['2015-01-01':'2015-12-31']['MS Return'],bins=100,color='green')
sns.distplot(returns.loc['2008-01-01':'2008-12-31']['C Return'],bins=100,color='red')
import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



# Optional Plotly Method Imports

from plotly import __version__
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)
bank_stocks.xs('Close',axis=1,level=1).plot(figsize=(12,5))
bank_stocks.loc['2008-01-01':'2008-12-31'].xs('Close',axis=1,level=1)['BAC'].plot(figsize=(12,6),label='BofA Close')

rolling_avg = pd.DataFrame()

rolling_avg['30 Day Avg'] = bank_stocks.loc['2008-01-01':'2008-12-31'].xs('Close',axis=1,level=1)['BAC'].rolling(window=30).mean()

rolling_avg['30 Day Avg'].plot(figsize=(12,6),label='30 Day Avg')

plt.legend()
sns.heatmap(bank_stocks.xs('Close',axis=1,level=1).corr(),annot=True,cmap='Blues')
sns.clustermap(bank_stocks.xs('Close',axis=1,level=1).corr(),annot=True,cmap='Blues')