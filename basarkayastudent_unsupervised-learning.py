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
data=pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
data
f,ax=plt.subplots(figsize=(14,7))

plt.scatter(data.trestbps, data.chol)

plt.show()
f,ax=plt.subplots(figsize=(14,7))

plt.scatter(data.trestbps, data.thalach)

plt.show()
data1=pd.DataFrame({"trestbps": data.trestbps, "thalach": data.thalach, "target": data.target})
data1
data2=data1.drop(["target"], axis=1)
from sklearn.cluster import KMeans



wcss=[]

for each in range(1,15):

    kmeans2=KMeans(n_clusters=each)

    kmeans2.fit(data2)

    wcss.append(kmeans2.inertia_)

f,ax=plt.subplots(figsize=(10,5))

plt.plot([*range(1,15)], wcss)

plt.show()
kmeans=KMeans(n_clusters=2)

clusters=kmeans.fit_predict(data2)

data1["kmeans"]=clusters
clusters
plt.scatter(data1.trestbps[data1.kmeans==0], data1.thalach[data1.kmeans==0])

plt.scatter(data1.trestbps[data1.kmeans==1], data1.thalach[data1.kmeans==1])

plt.show()
plt.scatter(data.trestbps[data.target==1], data.thalach[data.target==1])

plt.scatter(data.trestbps[data.target==0], data.thalach[data.target==0])

plt.show()
data1
success_kmeans=[]

for i in range(data.shape[0]):

    if (data1.target[i]==data1.kmeans[i]):

        success_kmeans.append(1)

    else:

        success_kmeans.append(0)
print("Success Rate:", 100*np.mean(success_kmeans), "%")
from scipy.cluster.hierarchy import linkage,dendrogram

merg=linkage(data2, method="ward")

dendrogram(merg, leaf_rotation=90)

plt.show()
from sklearn.cluster import AgglomerativeClustering

hc=AgglomerativeClustering(n_clusters=2, linkage="ward", affinity="euclidean")

clusters=hc.fit_predict(data1)

data1["hier"]=clusters
data1
plt.scatter(data1.trestbps[data1.hier==0], data1.thalach[data1.hier==0])

plt.scatter(data1.trestbps[data1.hier==1], data1.thalach[data1.hier==1])

plt.show()
plt.scatter(data1.trestbps[data1.target==1], data1.thalach[data1.target==1])

plt.scatter(data1.trestbps[data1.target==0], data1.thalach[data1.target==0])

plt.show()
success_hier=[]

for i in range(data.shape[0]):

    if (data1.hier[i]==data1.target[i]):

        success_hier.append(1)

    else:

        success_hier.append(0)
print("Hierarchical Clustering Success:", 100*np.mean(success_hier), "%")
plt.scatter(data.trestbps[data.target==0], data.thalach[data.target==0], color="green", label="0")

plt.scatter(data.trestbps[data.target==1], data.thalach[data.target==1], color="red", label="1")

plt.title("Data")

plt.xlabel("trestbps")

plt.ylabel("thalach")

plt.legend()

plt.show()
plt.scatter(data1.trestbps[data1.kmeans==0], data1.thalach[data1.kmeans==0], color="green", label="0")

plt.scatter(data1.trestbps[data1.kmeans==1], data1.thalach[data1.kmeans==1], color="red", label="1")

plt.title("KMeans Clustering")

plt.xlabel("trestbps")

plt.ylabel("thalach")

plt.legend()

plt.show()

print("Success Rate:", 100*np.mean(success_kmeans), "%")
plt.scatter(data1.trestbps[data1.hier==0], data1.thalach[data1.hier==0], color="green", label="0")

plt.scatter(data1.trestbps[data1.hier==1], data1.thalach[data1.hier==1], color="red", label="1")

plt.title("Hierarchical Clustering")

plt.xlabel("trestbps")

plt.ylabel("thalach")

plt.legend()

plt.show()

print("Success Rate:", 100*np.mean(success_hier), "%")