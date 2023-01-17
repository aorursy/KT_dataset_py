# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")

data

data.head()
# As you can see there is no labels in data

# we need to import matplot library.

import matplotlib.pyplot as plt

plt.scatter(data["chol"],data["thalach"])

plt.xlabel("chol")

plt.ylabel("thalach")

plt.show()
# KMeans Clustering

data2 = data.loc[:,["chol","thalach"]]

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 2) # we choose 2 cluster in our data.

kmeans.fit(data2)

labels = kmeans.predict(data2)

plt.scatter(data["chol"],data["thalach"],c = labels)

plt.xlabel("chol")

plt.xlabel("thalach")

plt.show()
# cross tabulation table

df = pd.DataFrame({'labels':labels,"target":data["target"]})

ct = pd.crosstab(df['labels'],df["target"])

print(ct)
# inertia

inertia_list = np.empty(14)

for i in range(1,14):

    kmeans = KMeans(n_clusters=i)

    kmeans.fit(data2)

    inertia_list[i] = kmeans.inertia_

plt.plot(range(0,14),inertia_list,"-o")

plt.xlabel("Number of cluster")

plt.ylabel("Inertia")

plt.show()
data = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")

data3 = data.drop("target",axis = 1)
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

scalar = StandardScaler()

kmeans = KMeans(n_clusters = 2)

pipe = make_pipeline(scalar,kmeans)

pipe.fit(data3)

labels = pipe.predict(data3)

df = pd.DataFrame({"labels":labels,"target":data["target"]})

ct = pd.crosstab(df["labels"],df["target"])

print(ct)
from scipy.cluster.hierarchy import linkage, dendrogram



merg = linkage(data,method = "ward")

dendrogram(merg,leaf_rotation = 90)

plt.show()
# HC

from sklearn.cluster import AgglomerativeClustering

hiyerartical_cluster = AgglomerativeClustering(n_clusters = 2,affinity="euclidean", linkage = "ward")

cluster = hiyerartical_cluster.fit_predict(data)



data["label"] = cluster

plt.scatter(data.chol[data.label == 0 ],data.thalach[data.label == 0],color = "red")

plt.scatter(data.chol[data.label == 1 ],data.thalach[data.label == 1],color = "green")

plt.show()