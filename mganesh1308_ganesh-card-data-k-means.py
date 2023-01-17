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
dataf=pd.read_csv("../input/CreditCardUsage.csv")
dataf.head()
dataf.info()
dataf.shape
dataf.isna().sum()
dataf.duplicated().sum()
dataf[dataf["PAYMENTS"]==0.00].shape
dataf["PAYMENTS"].value_counts()
dataf[dataf["PAYMENTS"]==0.00].shape
dataf["MINIMUM_PAYMENTS"].nunique()
import matplotlib.pyplot as plt 

from sklearn.cluster import KMeans 

from sklearn.datasets.samples_generator import make_blobs 

import pylab as pl

%matplotlib inline

from matplotlib import pyplot

from sklearn import cluster
data=dataf.drop(columns='CUST_ID')
data=data.fillna(0.00)
data.isna().sum()
df = dataf.drop('CUST_ID', axis=1)

df.head()
from sklearn.preprocessing import StandardScaler

X = df.values[:,1:]

X = np.nan_to_num(X)

Clus_dataSet = StandardScaler().fit_transform(X)

Clus_dataSet
clusterNum = 7

k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)

k_means.fit(X)

labels = k_means.labels_

print(labels)
Nc = range(1, 20)

kmeans = [KMeans(n_clusters=i) for i in Nc]

kmeans

score = [kmeans[i].fit(X).score(X) for i in range(len(kmeans))]

score

pl.plot(Nc,score)

pl.xlabel('Number of Clusters')

pl.ylabel('Score')

pl.title('Elbow Curve')

pl.show()
Nc = range(1, 20)

kmeans = [KMeans(n_clusters=i) for i in Nc]

kmeans

score = [kmeans[i].fit(X).inertia_ for i in range(len(kmeans))]

score

pl.plot(Nc,score)

pl.xlabel('Number of Clusters')

pl.ylabel('Sum of within sum square')

pl.title('Elbow Curve')

pl.show()
df["Clus_km"] = labels

df.head(5)
df["Clus_km"] = labels

df.head(5)
df.groupby('Clus_km').mean()

list(labels)
import scipy.cluster.hierarchy as sch



plt.figure(figsize=(15,6))

plt.title('Dendrogram')

plt.xlabel('Customers')

plt.ylabel('Euclidean distances')

#plt.grid(True)

dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))

plt.show()