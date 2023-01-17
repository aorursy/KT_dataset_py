# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import glob

filelist = glob.glob("../input/*.csv")



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd 

import glob

import seaborn as sns

from sklearn.cluster import AgglomerativeClustering

import matplotlib



import matplotlib.pyplot as plt



%matplotlib inline 



plt.rcParams['figure.figsize'] = (15, 10)

plt.rcParams.update({'font.size': 17})




stockDf = pd.DataFrame()



for i in filelist:

    tmp = pd.read_csv(i)

    tmp['symbol'] = i.split('/')[-1].split('.')[0]

    stockDf = stockDf.append(tmp)



stockDf.head()



### 1. use close price to calculate return 

### 2. use return to calculate correlation matrix

### 3. heatmap to visualize correlation matrix



stockDfClose = stockDf[['date', 'close', 'symbol']]

stockDfClose = stockDfClose.pivot(index='date', columns='symbol', values='close')

stockDfRet = np.log(stockDfClose/stockDfClose.shift(1))

stockDfRet = stockDfRet.dropna()

stockDfRetCorr = stockDfRet.corr()



ax = sns.heatmap(stockDfRetCorr)





### 1. calcualte the distane between stocks (This is common in finance. np.sqrt(2 * (1 - /rho)))

### 2. agglomerative clustering by distance matrix (use defualt euclidean distance and ward linkage)



dist = np.sqrt(2 * (1 - stockDfRetCorr))



clustering = AgglomerativeClustering(n_clusters=3).fit(dist)

stocksyms = dist.index.values



stock_clusters = pd.DataFrame({'Symbol':stocksyms, 'Cluster':clustering.labels_})



stock_clusters



cls0 = stock_clusters[stock_clusters['Cluster'] == 0]['Symbol']

cls1 = stock_clusters[stock_clusters['Cluster'] == 1]['Symbol']

cls2 = stock_clusters[stock_clusters['Cluster'] == 2]['Symbol']



cls0RetDf = stockDfRet.loc[:,cls0]

cls1RetDf = stockDfRet.loc[:,cls1]

cls2RetDf = stockDfRet.loc[:,cls2]



cls0Ret = cls0RetDf.mean(axis=1)

cls1Ret = cls1RetDf.mean(axis=1)

cls2Ret = cls2RetDf.mean(axis=1)



cls0Ret.cumsum().plot()

cls1Ret.cumsum().plot()

cls2Ret.cumsum().plot()

stockDfRet.mean(axis=1).cumsum().plot()

plt.legend(['Cluster 1', 'Cluster 2', 'Cluster 3', 'Dow Jone Index'])
