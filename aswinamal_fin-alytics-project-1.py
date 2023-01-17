#regular chore
import pandas as pd
import numpy as np
import datetime
import matplotlib 
from matplotlib import pyplot
%matplotlib inline
stock_info = pd.read_pickle('../input/all_banks')
#The tickers here is simply used to indicate the data values pertaining to each bank
tickers = ['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC']
stock_info.head()
stock_info.xs('Close', level=1, axis=1).max()
returns = pd.DataFrame()
for x in tickers:
    returns[x + ' Return'] = stock_info.xs('Close',axis=1,level=1)[x].pct_change()
returns.head()
import seaborn as sns
sns.set(style="ticks", color_codes=True)
sns.pairplot(returns)
stock_info.xs('Close',axis=1,level=1).plot(figsize=(20,10))
stock_info.xs('Volume',axis=1,level=1).plot(figsize=(20,10))
returns.std() 
returns.loc['2007-01-01':'2009-12-31'].std()
fig, ax = pyplot.subplots(figsize=(15,8))

sns.distplot(returns.loc['2007-01-01':'2009-12-31']['C Return'],color='red',ax=ax)

fig, ax = pyplot.subplots(figsize=(15,8))

sns.heatmap(stock_info.xs('Close',axis=1,level=1).corr(),annot=True,cmap='Reds',ax=ax)
sns.clustermap(stock_info.xs('Close',axis=1,level=1).corr(),annot=True,cmap='Blues')