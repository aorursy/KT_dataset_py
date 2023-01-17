

import pandas as pd # data processing

import numpy as np

import matplotlib.pyplot as plt

#%matplotlib inline



import os

from sklearn.cluster import KMeans

from scipy.cluster.hierarchy import linkage,dendrogram

import pandas as pd

data = pd.read_csv("../input/heart-disease-uci/heart.csv")
t = data.head(10)

print (t)

data["target"].unique()
data.info()

plt.scatter(data['age'],data['chol'])

plt.xlabel("age")

plt.ylabel("chol")

plt.show()
#KMeans Clustering

data2=data.loc[:,["age","chol"]]



kmeans=KMeans(n_clusters=2)

kmeans.fit(data2)



labels=kmeans.predict(data2)

plt.scatter(data["age"],data["chol"],c=labels)

plt.xlabel("age")

plt.ylabel("chol")

plt.show()

#inertia

inertia_list=np.empty(9)

for i in range(1,9):

    kmeans=KMeans(n_clusters=i)

    kmeans.fit(data2)

    inertia_list[i]=kmeans.inertia_



plt.plot(range(0,9),inertia_list)

plt.xlabel("Number of cluster")

plt.ylabel("İnertia")

plt.show()

data3=data.drop("target",axis=1)

#Hierarchy

from scipy.cluster.hierarchy import linkage,dendrogram



merg=linkage(data3.iloc[200:220,:],method="single")

dendrogram(merg,leaf_rotation=90,leaf_font_size=6)

plt.show()