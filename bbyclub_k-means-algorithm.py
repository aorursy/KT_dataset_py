# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/musteriler.csv") #import data

data.head()
X=data.iloc[:,3:].values
from sklearn.cluster import KMeans

kmeans= KMeans(n_clusters=3, init="k-means++")

kmeans.fit(X)



print(kmeans.cluster_centers_)

sonuclar=[]

for i in range(1,11):

    kmeans= KMeans(n_clusters=i, init= "k-means++", random_state=42)

    kmeans.fit(X)

    sonuclar.append(kmeans.inertia_) #give us WSCC values



plt.plot(range(1,11),sonuclar) #draw and find elbow point

plt.show()



kmeans= KMeans(n_clusters=4, init= "k-means++", random_state=42)

Y_tahmin=kmeans.fit_predict(X)

print(Y_tahmin)

plt.scatter(X[Y_tahmin==0,0], X[Y_tahmin==0,1], s=100, c="red")

plt.scatter(X[Y_tahmin==1,0], X[Y_tahmin==1,1], s=100, c="blue")

plt.scatter(X[Y_tahmin==2,0], X[Y_tahmin==2,1], s=100, c="green")

plt.scatter(X[Y_tahmin==3,0], X[Y_tahmin==3,1], s=100, c="yellow")

plt.title("KMeans")

plt.show()
from sklearn.cluster import AgglomerativeClustering

ac= AgglomerativeClustering(n_clusters=3, affinity="euclidean", linkage="ward")

Y_tahmin=ac.fit_predict(X)

print(Y_tahmin)



plt.scatter(X[Y_tahmin==0,0], X[Y_tahmin==0,1], s=100, c="red")

plt.scatter(X[Y_tahmin==1,0], X[Y_tahmin==1,1], s=100, c="blue")

plt.scatter(X[Y_tahmin==2,0], X[Y_tahmin==2,1], s=100, c="green")

plt.title("Hierarchical Clustering")

plt.show()
import scipy.cluster.hierarchy as sch

dendrogram= sch.dendrogram(sch.linkage(X, method="ward"))

plt.show()