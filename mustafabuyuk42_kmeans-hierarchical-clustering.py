# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Wholesale customers data.csv')
data.head()
data['Region'].value_counts()

data['Channel'].value_counts()
data.info()
X = data.iloc[:,2:8].values
X = (X-np.min(X))/(np.max(X)-np.min(X))
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x = sc.fit_transform(X)
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 3, init = 'k-means++')

kmeans.fit(X)



results =[]

for i in range(1,15):

    kmeans2 = KMeans(n_clusters = i, init = 'k-means++',random_state=123)

    kmeans2.fit(X)

    results.append(kmeans2.inertia_)



plt.figure(figsize = (15,8))



plt.plot(range(1,15),results)

plt.xlabel('Number of K')

plt.ylabel('Inertia')

plt.show()
y_kmeans_pred = kmeans.predict(X)

plt.figure(figsize = (12,6))

plt.scatter(X[y_kmeans_pred == 0,0],X[y_kmeans_pred == 0,1],color = 'red')

plt.scatter(X[y_kmeans_pred == 1,0],X[y_kmeans_pred == 1,1],color = 'green')

plt.scatter(X[y_kmeans_pred == 2,0],X[y_kmeans_pred == 2,1],color = 'blue')



plt.title('Kmeans')

plt.show()
from sklearn.cluster import AgglomerativeClustering

agc = AgglomerativeClustering(n_clusters=3,affinity = 'euclidean',linkage = 'ward')

y_agc_pred = agc.fit_predict(X)

plt.figure(figsize =(12,8))

plt.scatter(X[y_agc_pred == 0,0],X[y_agc_pred ==0,1],color= 'red')

plt.scatter(X[y_agc_pred == 1,0],X[y_agc_pred ==1,1],color= 'green')

plt.scatter(X[y_agc_pred == 2,0],X[y_agc_pred ==2,1],color= 'blue')

plt.title('Agglomerative Clustering')

plt.show()
from scipy.cluster import hierarchy 

Z = hierarchy.linkage(X,method = 'ward')



dendogram =hierarchy.dendrogram(Z)



plt.show()

Z2= hierarchy.linkage(X,method = 'centroid')

plt.figure(figsize=(15,5))

dendogram2 =hierarchy.dendrogram(Z2)

plt.show()