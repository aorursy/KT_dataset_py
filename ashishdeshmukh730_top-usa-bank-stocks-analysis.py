import pandas as pd

import numpy as np

import matplotlib as mt

import seaborn as sns

import seaborn as sns

sns.set_style('whitegrid')

sns.set()

import scipy 

import matplotlib.pyplot as plt

import datetime

%matplotlib inline

import pickle

import pprint

import json



from sklearn.preprocessing import StandardScaler



from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.cluster import KMeans
BOA = pd.read_csv("../input/bank-stock-data/Bank of America.csv")

C = pd.read_csv("../input/bank-stock-data/City Group.csv")

GS = pd.read_csv("../input/bank-stock-data/Goldman Sachs.csv")

JPM = pd.read_csv("../input/bank-stock-data/JP Morgan.csv")

MS = pd.read_csv("../input/bank-stock-data/Morgan Stanley.csv")

WFC = pd.read_csv("../input/bank-stock-data/Wells Fargo.csv")

bank_stocks = pd.read_pickle("../input/all-banks/all_banks")

tickers = ['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC']

# first 10 records 

bank_stocks.head(10)
# Descriptive statistics 



bank_stocks.describe()
#correlation (volume)

bank_stocks.xs('Volume',axis=1,level=1).corr()
#Correlation heat map (Volume)

x_axis_labels = ['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC']

sns.heatmap(bank_stocks.xs('Volume',axis=1,level=1).corr(),annot=True,cmap='Blues',xticklabels=x_axis_labels)

plt.ylim(6.0, 0)
#correlation (Closing Price)

bank_stocks.xs('Close',axis=1,level=1).corr()
#Correlation heat map (Closing price)

x_axis_labels = ['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC']

sns.heatmap(bank_stocks.xs('Close',axis=1,level=1).corr(),annot=True,cmap='Blues',xticklabels=x_axis_labels)

plt.ylim(6.0, 0)

#correlation (opening price)

bank_stocks.xs('Open',axis=1,level=1).corr()
#Correlation heat map (opening price)

plt.figure(figsize = (6.0,6.0))

s =sns.heatmap(bank_stocks.xs('Open',axis=1,level=1).corr(), 

            vmin = -1,                       

            vmax = 1,                      

            cmap = 'Blues',                      

            annot = True)

s.set_yticklabels(s.get_yticklabels(), rotation = 0, fontsize = 12)

s.set_xticklabels(s.get_xticklabels(), rotation = 90, fontsize = 12)

plt.ylim(6.0, 0)


hier_clust = linkage(bank_stocks.xs('Volume',axis=1,level=1).corr(), method = 'ward')



plt.figure(figsize = (12,9))

plt.title('Hierarchical Clustering Dendrogram')

plt.xlabel('Observations')

plt.ylabel('Distance')

dendrogram(hier_clust,

           truncate_mode = 'level',

           p = 5,

           show_leaf_counts = False,

           no_labels = True)

plt.show()
# Cluster map (Closing map)

sns.clustermap(bank_stocks.xs('Close',axis=1,level=1).corr(),annot=True,cmap='Blues')
bank_stocks.xs('Close', level=1, axis=1).max()



returns = pd.DataFrame()



for x in tickers:

    returns[x + ' Return'] = bank_stocks.xs('Close',axis=1,level=1)[x].pct_change()

returns.head()
sns.pairplot(returns[1:])
returns.idxmax()
returns.idxmin()
# over all risk 

returns.std() 
# Risk analysis post-financial crisis

returns.loc['2015-01-01':'2015-12-31'].std()



bank_stocks.xs('Close',axis=1,level=1).plot(figsize=(12,5))
bank_stocks.loc['2008-01-01':'2008-12-31'].xs('Close',axis=1,level=1)['BAC'].plot(figsize=(12,6),label='BofA Close')

rolling_avg = pd.DataFrame()

rolling_avg['30 Day Avg'] = bank_stocks.loc['2008-01-01':'2008-12-31'].xs('Close',axis=1,level=1)['BAC'].rolling(window=30).mean()

rolling_avg['30 Day Avg'].plot(figsize=(12,6),label='30 Day Avg')

plt.legend()