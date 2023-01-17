# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("/kaggle/input/iris/Iris.csv")
data
data.Species.value_counts()
data.Species = data.Species.replace('Iris-setosa',1)



data.Species = data.Species.replace('Iris-versicolor',2)



data.Species = data.Species.replace('Iris-virginica',3)
Iris_setosa = data[data.Species == 1]



Iris_versicolor = data[data.Species==2]



Iris_virginica = data[data.Species==3]

plt.scatter(Iris_setosa["PetalLengthCm"],Iris_setosa["SepalWidthCm"],color = "red",alpha=0.8)

plt.scatter(Iris_versicolor["PetalLengthCm"],Iris_versicolor["SepalWidthCm"],color="green",alpha=0.8)

plt.scatter(Iris_virginica["PetalLengthCm"],Iris_virginica["SepalWidthCm"],color = "blue",alpha=0.8)

plt.xlabel("PetalLengthCm")

plt.ylabel("PetalWidthCm")

plt.legend()

plt.show()

plt.scatter(Iris_setosa["PetalLengthCm"],Iris_setosa["SepalWidthCm"],color="black",alpha=1)

plt.scatter(Iris_versicolor["PetalLengthCm"],Iris_versicolor["SepalWidthCm"],color = "black",alpha=1)

plt.scatter(Iris_virginica["PetalLengthCm"],Iris_virginica["SepalWidthCm"],color = "black",alpha=1)

plt.xlabel("PetalLengthCm")

plt.ylabel("PetalWidthCm")

plt.legend()

plt.show()
from sklearn.cluster import KMeans

wcss=[]



for k in range(1, 15):

    kmeans=KMeans(n_clusters=k)

    kmeans.fit(data)

    wcss.append(kmeans.inertia_)



kmeans2 =KMeans(n_clusters=3)



clusters = kmeans2.fit_predict(data)



data["label"] = clusters



plt.scatter(data["PetalLengthCm"][data.label==0],data["SepalWidthCm"][data.label==0], color="red")

plt.scatter(data["PetalLengthCm"][data.label==1],data["SepalWidthCm"][data.label==1], color="green")

plt.scatter(data["PetalLengthCm"][data.label==2],data["SepalWidthCm"][data.label==2], color="blue")



plt.show()
from sklearn.cluster import KMeans

wcss=[]



for k in range(1, 15):

    kmeans=KMeans(n_clusters=k)

    kmeans.fit(data)

    wcss.append(kmeans.inertia_)

    

plt.plot(range(1,15),wcss)

plt.xlabel("number of k (cluster) value")

plt.xlabel("wcss")

plt.show()
from scipy.cluster.hierarchy import linkage, dendrogram



merg = linkage(data,method="ward")

dendrogram(merg,leaf_rotation = 90)

plt.xlabel("data points")

plt.ylabel("euclidean distance")

plt.show()
from sklearn.cluster import AgglomerativeClustering

hiyerartical_cluster = AgglomerativeClustering(n_clusters = 2,affinity ="euclidean",linkage="ward")



cluster=hiyerartical_cluster.fit_predict(data)



data["label"] = cluster



plt.scatter(data["PetalLengthCm"][data.label==0],data["SepalWidthCm"][data.label==0], color="red")

plt.scatter(data["PetalLengthCm"][data.label==1],data["SepalWidthCm"][data.label==1], color="yellow")



plt.show()