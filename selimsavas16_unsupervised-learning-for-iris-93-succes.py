# Mathematics
import numpy as np
import pandas as pd 

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

data = pd.read_csv("../input/iris/Iris.csv")
data.head()
data.tail()
data.info()
data2 = data.drop('Species',axis = 1)
data2
data2.info()
data2.corr()
from sklearn.cluster import KMeans

wcss = []


for k in range(1,15):
    
    Kmeans = KMeans(n_clusters=k)
    Kmeans.fit(data2)
    wcss.append(Kmeans.inertia_)
    
    
plt.plot(range(1,15),wcss)
plt.xlabel("number of k values")
plt.ylabel("wcss")
plt.show()
# k=3 için model

kmeans2 = KMeans(n_clusters = 3)
clusters = kmeans2.fit_predict(data2)
data2["label"]= clusters
data2
data2["label"].unique()
plt.scatter(data2.PetalWidthCm[data2.label==0],data2.PetalLengthCm[data2.label==0], color="red")
plt.scatter(data2.PetalWidthCm[data2.label==1],data2.PetalLengthCm[data2.label==1], color="green")
plt.scatter(data2.PetalWidthCm[data2.label==2],data2.PetalLengthCm[data2.label==2], color="blue")
plt.scatter(x="PetalWidthCm",y="PetalLengthCm",c="label", data=data2)
plt.show()
from scipy.cluster.hierarchy import linkage,dendrogram

merg = linkage(data2,method="ward")
dendrogram(merg,leaf_rotation=90)

plt.xlabel("Data Points")
plt.ylabel("Euvliden distance")
plt.show()
from sklearn.cluster import AgglomerativeClustering

h_c = AgglomerativeClustering(n_clusters=3,linkage="ward")

cluster = h_c.fit_predict(data2)

data2["label"] = cluster
data2["label"].unique()
kmeans2.cluster_centers_[:,4] # x_centers
kmeans2.cluster_centers_[:,3] # y_centers
plt.scatter(data2.PetalWidthCm[data2.label==0],data2.PetalLengthCm[data2.label==0], color="red")
plt.scatter(data2.PetalWidthCm[data2.label==1],data2.PetalLengthCm[data2.label==1], color="green")
plt.scatter(data2.PetalWidthCm[data2.label==2],data2.PetalLengthCm[data2.label==2], color="blue")
plt.scatter(kmeans2.cluster_centers_[:,4],kmeans2.cluster_centers_[:,3], color="yellow")
plt.show()
data
Species_Values = []

for each in data.Species:
    
    if (each == "Iris-setosa"):
        Species_Values.append(0)
        
    elif(each == "Iris-versicolor"):
        Species_Values.append(2)
        
    else:
        Species_Values.append(1)

data["Species"] = Species_Values
data.head()
data["Guess"] = cluster
data.head()
p = data.shape[0]
Success = []
for i in np.arange(p):
    
    if( data.loc[i,"Species"] == data.loc[i,"Guess"]):
        Success.append(1)
    else:
        Success.append(0)
    
data["Success"] = Success
data
# data.shape[0] : 150
print("Cluster Success : {}".format(data["Success"].sum()/data.shape[0]))
