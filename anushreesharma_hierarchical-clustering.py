# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import scipy.cluster.hierarchy as sch

from sklearn.cluster import AgglomerativeClustering



%matplotlib inline

df = pd.read_csv('../input/Mall_Customers.csv')

df.head(10)

df.describe()



X = df.iloc[:,[3,4]].values

plt.figure(figsize=(15,6))

plt.title('Dendrogram')

plt.xlabel('Customers')

plt.ylabel('Euclidean distances')

#plt.grid(True)

dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))

plt.show()



plt.figure(figsize=(15,6))

plt.title('Dendrogram')

plt.xlabel('Customers')

plt.ylabel('Euclidean distances')

plt.hlines(y=190,xmin=0,xmax=2000,lw=3,linestyles='--')

plt.text(x=900,y=220,s='Horizontal line crossing 5 vertical lines',fontsize=20)

#plt.grid(True)

dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))

plt.show()



hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')

y_hc = hc.fit_predict(X)

plt.figure(figsize=(12,7))

plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Careful-Cluster1')

plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Standard-Cluster2')

plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Target group-Cluster3')

plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'orange', label = 'Careless-Cluster4')

plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Sensible-Cluster5')

plt.title('Clustering of customers',fontsize=20)

plt.xlabel('Annual Income (k$)',fontsize=16)

plt.ylabel('Spending Score (1-100)',fontsize=16)

plt.legend(fontsize=16)

plt.grid(True)

plt.axhspan(ymin=60,ymax=100,xmin=0.4,xmax=0.96,alpha=0.3,color='yellow')

plt.show()
