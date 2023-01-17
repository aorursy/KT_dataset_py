# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
x1 = np.random.normal(20,5,1000)
y1 = np.random.normal(20,5,1000)

x2 = np.random.normal(60,5,1000)
y2 = np.random.normal(20,5,1000)

x3 = np.random.normal(40,5,1000)
y3 = np.random.normal(40,5,1000)

x = np.concatenate([x1,x2,x3], axis=0)
y = np.concatenate([y1,y2,y3], axis=0)

data = {"x":x,"y":y}
data = pd.DataFrame(data)

plt.scatter(x1,y1,color="black")
plt.scatter(x2,y2,color="black")
plt.scatter(x3,y3,color="black")
plt.show()

data.head()
wcss = []
for k in range(1,15):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,15),wcss)
plt.xlabel("number of k value")
plt.ylabel("wcss")
plt.show()
kmeans2 = KMeans(n_clusters=3)
clusters = kmeans2.fit_predict(data)
data["label"] = clusters

plt.scatter(data.x[data.label == 0], data.y[data.label == 0])
plt.scatter(data.x[data.label == 1], data.y[data.label == 1])
plt.scatter(data.x[data.label == 2], data.y[data.label == 2])
plt.show()
merg = linkage(data, method="ward")
dendrogram(merg)
plt.xlabel("data points")
plt.ylabel("euclidean distance")
plt.show()
hiyerartical_cluster = AgglomerativeClustering(n_clusters=3, affinity="euclidean", linkage="ward")
cluster = hiyerartical_cluster.fit_predict(data)
data["label2"] = cluster

plt.scatter(data.x[data.label2 == 0], data.y[data.label2 == 0])
plt.scatter(data.x[data.label2 == 1], data.y[data.label2 == 1])
plt.scatter(data.x[data.label2 == 2], data.y[data.label2 == 2])
plt.show()








