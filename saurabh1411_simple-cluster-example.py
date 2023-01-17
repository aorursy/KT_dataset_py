# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataset=pd.read_csv('../input/clustering/musteriler.csv')
dataset.head()
X=dataset.iloc[:,3:].values

X
import matplotlib.pyplot as plt 

plt.scatter(X[:,0],X[:,1])
from sklearn.cluster import KMeans

neg=[]

for i in range(1,11):

    kmeans=KMeans(n_clusters = i, init='k-means++', random_state= 123)

    kmeans.fit(X)

    neg.append(kmeans.inertia_)

plt.plot(range(1,11),clusters)
kmeans=KMeans(n_clusters=4,init='k-means++')

kmeans.fit(X)
clusters = kmeans.fit_predict(X)

dataset["label"] = clusters
dataset
plt.scatter(dataset.Hacim[dataset.label == 0 ],dataset.Maas[dataset.label == 0],color = "red")

plt.scatter(dataset.Hacim[dataset.label == 1 ],dataset.Maas[dataset.label == 1],color = "green")

plt.scatter(dataset.Hacim[dataset.label == 2 ], dataset.Maas[dataset.label == 2],color = "blue")

plt.scatter(dataset.Hacim[dataset.label == 3 ], dataset.Maas[dataset.label == 3],color = "violet")

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color = "yellow")

plt.show()
from scipy.cluster.hierarchy import linkage, dendrogram



merg = linkage(X,method="ward")

dendrogram(merg,leaf_rotation = 90)

plt.xlabel("data points")

plt.ylabel("euclidean distance")

plt.show()
from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')

Y_tahmin = ac.fit_predict(X)

#print(Y_tahmin)



plt.scatter(X[Y_tahmin==0,0],X[Y_tahmin==0,1],s=100, c='red')

plt.scatter(X[Y_tahmin==1,0],X[Y_tahmin==1,1],s=100, c='blue')

plt.scatter(X[Y_tahmin==2,0],X[Y_tahmin==2,1],s=100, c='green')

plt.scatter(X[Y_tahmin==3,0],X[Y_tahmin==3,1],s=100, c='yellow')

plt.title('HC')

plt.show()