# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import warnings

warnings.filterwarnings("ignore")



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#create dataset

#np.random.normal(25,5,1000) =>

#it's mean that 66%(666) of my data will be between 20(25-5) and 30(25+5)

#This distribution is called Gaussian distribution.



#for class1

x1 = np.random.normal(25,5,1000)

y1 = np.random.normal(25,5,1000)



#for class2

x2 = np.random.normal(55,5,1000)

y2 = np.random.normal(60,5,1000)



#for class3

x3 = np.random.normal(55,5,1000)

y3 = np.random.normal(15,5,1000)



x = np.concatenate((x1,x2,x3), axis=0)

y = np.concatenate((y1,y2,y3), axis=0)
dictionary = {"x":x,"y":y}

data = pd.DataFrame(dictionary)
#my data look like this



plt.scatter(x1,y1)

plt.scatter(x2,y2)

plt.scatter(x3,y3)

plt.show()
#but kmeans algorithm will see this way



plt.scatter(x1,y1,color="black")

plt.scatter(x2,y2,color="black")

plt.scatter(x3,y3,color="black")

plt.show()
#K-MEANS

from sklearn.cluster import KMeans

wcss = []



for k in range(1,15):

    kmeans = KMeans(n_clusters=k)

    kmeans.fit(data)

    #kmeans.inertia_ =>find wcss for each key value and add to list

    wcss.append(kmeans.inertia_)



#as we can see,optimal point is k=3

plt.plot(range(1,15),wcss)

plt.xlabel("number of k (cluster) value")

plt.ylabel("wcss")

plt.show()
plt.figure(figsize=(20,10))

plt.suptitle("K Means Clustering",fontsize=20)



#This is my model for k = 2.

#fit_predict(data)=>it's mean that fit my data and create my clusters.

kmeans2 = KMeans(n_clusters=1)

cluster = kmeans2.fit_predict(data)

data["label"] = cluster

plt.subplot(1,3,1)

plt.scatter(data.x[data.label == 0],data.y[data.label == 0],color="red")

plt.scatter(data.x[data.label == 1],data.y[data.label == 1],color="green")

plt.scatter(kmeans2.cluster_centers_[:,0],kmeans2.cluster_centers_[:,1],color="black")



# drop labels to use for k=2

data.drop(["label"],axis=1,inplace=True)



kmeans2 = KMeans(n_clusters=2)

cluster = kmeans2.fit_predict(data)

data["label"] = cluster

plt.subplot(1,3,2)

plt.scatter(data.x[data.label == 0],data.y[data.label == 0],color="red")

plt.scatter(data.x[data.label == 1],data.y[data.label == 1],color="green")

plt.scatter(kmeans2.cluster_centers_[:,0],kmeans2.cluster_centers_[:,1],color="black")



# drop labels to use for k=3

data.drop(["label"],axis=1,inplace=True)



kmeans2 = KMeans(n_clusters=3)

cluster = kmeans2.fit_predict(data)

data["label"] = cluster

plt.subplot(1,3,3)

plt.scatter(data.x[data.label == 0],data.y[data.label == 0],color="red")

plt.scatter(data.x[data.label == 1],data.y[data.label == 1],color="green")

plt.scatter(data.x[data.label == 2],data.y[data.label == 2],color="blue")

plt.scatter(kmeans2.cluster_centers_[:,0],kmeans2.cluster_centers_[:,1],color="black")
# dendogram

from scipy.cluster.hierarchy import linkage, dendrogram



merg = linkage(data,method="ward")

plt.figure(figsize=(15,8))

dendrogram(merg,leaf_rotation = 90)

plt.xlabel("data points")

plt.ylabel("euclidean distance")

plt.show()
# HC

from sklearn.cluster import AgglomerativeClustering



hiyerartical_cluster = AgglomerativeClustering(n_clusters = 3,affinity= "euclidean",linkage = "ward")

cluster = hiyerartical_cluster.fit_predict(data)



data["label"] = cluster



plt.scatter(data.x[data.label == 0 ],data.y[data.label == 0],color = "red")

plt.scatter(data.x[data.label == 1 ],data.y[data.label == 1],color = "green")

plt.scatter(data.x[data.label == 2 ],data.y[data.label == 2],color = "blue")

plt.scatter(kmeans2.cluster_centers_[:,0],kmeans2.cluster_centers_[:,1],color="black")

plt.show()
data = pd.read_csv("../input/Iris.csv")
data.head()
#we drop Id column because it's not a useful feature for us.

data.drop(["Id"],axis=1,inplace=True)
data.info()
sns.pairplot(data=data,hue="Species",palette="Set1")

plt.show()
features = data.loc[:,["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]
#K-MEANS

from sklearn.cluster import KMeans

wcss = []



for k in range(1,15):

    kmeans = KMeans(n_clusters=k)

    kmeans.fit(features)

    #kmeans.inertia_ =>find wcss for each key value and add to list

    wcss.append(kmeans.inertia_)



#as we can see,optimal point is k=3

plt.figure(figsize=(10,8))

plt.plot(range(1,15),wcss,"-o")#"-o"=> for marker(point)

plt.title("WCSS-K Chart", fontsize=18)

plt.grid(True)

plt.xlabel("Number of K (cluster) Value")

plt.ylabel("WCSS")

plt.xticks(range(1,15))

plt.show()
plt.figure(figsize=(20,20))

plt.suptitle("K Means Clustering",fontsize=20)





plt.subplot(3,2,1)

plt.title("K = 1",fontsize=16)

plt.xlabel("PetalLengthCm")

plt.ylabel("PetalWidthCm")

plt.scatter(features.PetalLengthCm,features.PetalWidthCm)





plt.subplot(3,2,2)

plt.title("K = 2",fontsize=16)

plt.xlabel("PetalLengthCm")

kmeans = KMeans(n_clusters=2)

features["labels"] = kmeans.fit_predict(features)

plt.scatter(features.PetalLengthCm[features.labels == 0],features.PetalWidthCm[features.labels == 0])

plt.scatter(features.PetalLengthCm[features.labels == 1],features.PetalWidthCm[features.labels == 1])



# drop labels to use for k=3

features.drop(["labels"],axis=1,inplace=True)



plt.subplot(3,2,3)

plt.title("K = 3",fontsize=16)

plt.xlabel("PetalLengthCm")

kmeans = KMeans(n_clusters=3)

features["labels"] = kmeans.fit_predict(features)

plt.scatter(features.PetalLengthCm[features.labels == 0],features.PetalWidthCm[features.labels == 0])

plt.scatter(features.PetalLengthCm[features.labels == 1],features.PetalWidthCm[features.labels == 1])

plt.scatter(features.PetalLengthCm[features.labels == 2],features.PetalWidthCm[features.labels == 2])



# drop labels to use for k=4

features.drop(["labels"],axis=1,inplace=True)



plt.subplot(3,2,4)

plt.title("K = 4",fontsize=16)

plt.xlabel("PetalLengthCm")

kmeans = KMeans(n_clusters=4)

features["labels"] = kmeans.fit_predict(features)

plt.scatter(features.PetalLengthCm[features.labels == 0],features.PetalWidthCm[features.labels == 0])

plt.scatter(features.PetalLengthCm[features.labels == 1],features.PetalWidthCm[features.labels == 1])

plt.scatter(features.PetalLengthCm[features.labels == 2],features.PetalWidthCm[features.labels == 2])

plt.scatter(features.PetalLengthCm[features.labels == 3],features.PetalWidthCm[features.labels == 3])



# drop labels to use for k=5

features.drop(["labels"],axis=1,inplace=True)



plt.subplot(3,2,5)

plt.title("K = 5",fontsize=16)

plt.xlabel("PetalLengthCm")

kmeans = KMeans(n_clusters=5)

features["labels"] = kmeans.fit_predict(features)

plt.scatter(features.PetalLengthCm[features.labels == 0],features.PetalWidthCm[features.labels == 0])

plt.scatter(features.PetalLengthCm[features.labels == 1],features.PetalWidthCm[features.labels == 1])

plt.scatter(features.PetalLengthCm[features.labels == 2],features.PetalWidthCm[features.labels == 2])

plt.scatter(features.PetalLengthCm[features.labels == 3],features.PetalWidthCm[features.labels == 3])

plt.scatter(features.PetalLengthCm[features.labels == 4],features.PetalWidthCm[features.labels == 4])



# drop labels

features.drop(["labels"],axis=1,inplace=True)



plt.subplot(3,2,6)

plt.title("Original Labels",fontsize=16)

plt.xlabel("PetalLengthCm")

plt.scatter(data.PetalLengthCm[data.Species == "Iris-setosa"],data.PetalWidthCm[data.Species == "Iris-setosa"])

plt.scatter(data.PetalLengthCm[data.Species == "Iris-versicolor"],data.PetalWidthCm[data.Species == "Iris-versicolor"])

plt.scatter(data.PetalLengthCm[data.Species == "Iris-virginica"],data.PetalWidthCm[data.Species == "Iris-virginica"])



plt.show()

from scipy.cluster.hierarchy import dendrogram, linkage



merge = linkage(features,method="ward")



plt.figure(figsize=(15,8))

dendrogram(merge, leaf_rotation=90)

plt.xlabel("data points")

plt.ylabel("euclidian distance")

plt.suptitle("DENDROGRAM",fontsize=18)

plt.show()

# we draw a line on the longest vertical line between horizontal lines.

# Then we count how many vertical lines the line passes.

# As you can see,it's look like k=2 but we will see.
from sklearn.cluster import AgglomerativeClustering

plt.figure(figsize=(20,20))

plt.suptitle("Hierarchical Clustering",fontsize=20)





plt.subplot(3,2,1)

plt.title("K = 1",fontsize=16)

plt.xlabel("PetalLengthCm")

plt.ylabel("PetalWidthCm")

plt.scatter(features.PetalLengthCm,features.PetalWidthCm)





plt.subplot(3,2,2)

plt.title("K = 2",fontsize=16)

plt.xlabel("PetalLengthCm")

hc_cluster = AgglomerativeClustering(n_clusters=2)

features["labels"] = hc_cluster.fit_predict(features)

plt.scatter(features.PetalLengthCm[features.labels == 0],features.PetalWidthCm[features.labels == 0])

plt.scatter(features.PetalLengthCm[features.labels == 1],features.PetalWidthCm[features.labels == 1])



# drop labels to use for k=3

features.drop(["labels"],axis=1,inplace=True)



plt.subplot(3,2,3)

plt.title("K = 3",fontsize=16)

plt.xlabel("PetalLengthCm")

hc_cluster = AgglomerativeClustering(n_clusters=3)

features["labels"] = hc_cluster.fit_predict(features)

plt.scatter(features.PetalLengthCm[features.labels == 0],features.PetalWidthCm[features.labels == 0])

plt.scatter(features.PetalLengthCm[features.labels == 1],features.PetalWidthCm[features.labels == 1])

plt.scatter(features.PetalLengthCm[features.labels == 2],features.PetalWidthCm[features.labels == 2])



# drop labels to use for k=4

features.drop(["labels"],axis=1,inplace=True)



plt.subplot(3,2,4)

plt.title("K = 4",fontsize=16)

plt.xlabel("PetalLengthCm")

hc_cluster = AgglomerativeClustering(n_clusters=4)

features["labels"] = hc_cluster.fit_predict(features)

plt.scatter(features.PetalLengthCm[features.labels == 0],features.PetalWidthCm[features.labels == 0])

plt.scatter(features.PetalLengthCm[features.labels == 1],features.PetalWidthCm[features.labels == 1])

plt.scatter(features.PetalLengthCm[features.labels == 2],features.PetalWidthCm[features.labels == 2])

plt.scatter(features.PetalLengthCm[features.labels == 3],features.PetalWidthCm[features.labels == 3])



# drop labels to use for k=5

features.drop(["labels"],axis=1,inplace=True)



plt.subplot(3,2,5)

plt.title("K = 5",fontsize=16)

plt.xlabel("PetalLengthCm")

hc_cluster = AgglomerativeClustering(n_clusters=5)

features["labels"] = hc_cluster.fit_predict(features)

plt.scatter(features.PetalLengthCm[features.labels == 0],features.PetalWidthCm[features.labels == 0])

plt.scatter(features.PetalLengthCm[features.labels == 1],features.PetalWidthCm[features.labels == 1])

plt.scatter(features.PetalLengthCm[features.labels == 2],features.PetalWidthCm[features.labels == 2])

plt.scatter(features.PetalLengthCm[features.labels == 3],features.PetalWidthCm[features.labels == 3])

plt.scatter(features.PetalLengthCm[features.labels == 4],features.PetalWidthCm[features.labels == 4])



# drop labels

features.drop(["labels"],axis=1,inplace=True)



plt.subplot(3,2,6)

plt.title("Original Labels",fontsize=16)

plt.xlabel("PetalLengthCm")

plt.scatter(data.PetalLengthCm[data.Species == "Iris-setosa"],data.PetalWidthCm[data.Species == "Iris-setosa"])

plt.scatter(data.PetalLengthCm[data.Species == "Iris-versicolor"],data.PetalWidthCm[data.Species == "Iris-versicolor"])

plt.scatter(data.PetalLengthCm[data.Species == "Iris-virginica"],data.PetalWidthCm[data.Species == "Iris-virginica"])



plt.show()