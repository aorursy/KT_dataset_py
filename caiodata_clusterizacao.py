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
df = pd.read_csv("../input/states_all.csv", index_col=0)

df.head()
df.tail()
df.dropna(axis=0, inplace=True)
df.groupby('STATE')[['TOTAL_REVENUE', 'GRADES_ALL_G']].mean()
new_df = df[['TOTAL_REVENUE', 'GRADES_ALL_G']].copy()

new_df.head()
new_df.dropna(axis=0, inplace=True)
new_df.tail()
new_df.info()
new_df.plot.scatter(x='TOTAL_REVENUE', y='GRADES_ALL_G')
new_df2 = df.groupby('STATE')[['TOTAL_REVENUE', 'GRADES_ALL_G']].mean()

new_df2.head()
new_df2.plot.scatter(x='TOTAL_REVENUE', y='GRADES_ALL_G')
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage

from sklearn.cluster import AgglomerativeClustering

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
sc = StandardScaler()
base = sc.fit_transform(new_df2)

plt.scatter(x=base[:,0], y=base[:,1])
dendrograma = dendrogram(linkage(base, method='ward'))
hc =  AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
clusteres = hc.fit_predict(base)

clusteres
new_df2['cluster'] = clusteres
plt.scatter(x=new_df2.TOTAL_REVENUE.values, y=new_df2.GRADES_ALL_G, c=new_df2.cluster)
new_df2[new_df2.cluster == 1]
new_df2[new_df2.cluster == 2]
from sklearn.cluster import KMeans