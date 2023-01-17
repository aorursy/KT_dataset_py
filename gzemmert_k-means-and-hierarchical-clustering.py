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


# creating data 
#class1
x1= np.random.normal(25,5,1000)
y1= np.random.normal(25,5,1000)
#class2
x2= np.random.normal(55,5,1000)
y2= np.random.normal(60,5,1000)
#class3
x3= np.random.normal(55,5,1000)
y3= np.random.normal(15,5,1000)

x=np.concatenate((x1,x2,x3),axis=0)
y=np.concatenate((y1,y2,y3),axis=0)

dictionary={"x": x,"y":y}
data=pd.DataFrame(dictionary)

plt.scatter(x1,y1)
plt.scatter(x2,y2)
plt.scatter(x3,y3)
plt.show()


# K-Means Clustering

from sklearn.cluster import KMeans
wcss=[]

for k in range(1,15):
    kmeans=KMeans(n_clusters=k)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,15),wcss)
plt.xlabel("#k")
plt.ylabel("wcss")
plt.show()

#%% k =3 
kmeans2=KMeans(n_clusters=3)
clusters=kmeans2.fit_predict(data)
data["label"]=clusters

plt.scatter(data.x[data.label==0],data.y[data.label==0])
plt.scatter(data.x[data.label==1],data.y[data.label==1])
plt.scatter(data.x[data.label==2],data.y[data.label==2])
#to find the centers of clusters 
plt.scatter(kmeans2.cluster_centers_[:,0],kmeans2.cluster_centers_[:,1],color="yellow") 

plt.show()

#Hierarchical Clustering
from scipy.cluster.hierarchy import linkage,dendrogram

merg=linkage(data,method="ward") 
dendrogram(merg,leaf_rotation=90)
plt.xlabel("data points")
plt.ylabel("euclidean distance")
plt.show()

#3 cluster 

from sklearn.cluster import  AgglomerativeClustering
hc= AgglomerativeClustering(n_clusters=3,affinity="euclidean",linkage="ward")
cluster=hc.fit_predict(data)
data["label"]=cluster


plt.scatter(data.x[data.label==0],data.y[data.label==0])
plt.scatter(data.x[data.label==1],data.y[data.label==1])
plt.scatter(data.x[data.label==2],data.y[data.label==2])
plt.show
