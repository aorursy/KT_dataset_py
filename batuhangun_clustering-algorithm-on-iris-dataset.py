import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 

import warnings
warnings.filterwarnings("ignore")

import os
print(os.listdir("../input"))
data = pd.read_csv("../input/Iris.csv")
data.head()
X = data[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]
from sklearn.cluster import KMeans

l = []
for i in range(1,10):
    kmeans = KMeans(n_clusters=i,init="k-means++")
    cluster = kmeans.fit_predict(X)
    l.append(kmeans.inertia_)

plt.figure(figsize=(7,4))
sns.lineplot(x = range(1,10), y = l)
plt.show()
kmeans = KMeans(n_clusters=3,init="k-means++",random_state=47)
cluster = kmeans.fit_predict(X)
testClass = [1 if i =="Iris-setosa" else  0 if i =="Iris-versicolor" else 2  for i in data.Species]

count = 0
for i in range(len(cluster)):
    if cluster[i] == testClass[i]:
        count+=1
    
score = count/len(cluster)
print("K-Means Algorithm score: {}".format(score))
from scipy.cluster.hierarchy import dendrogram,linkage

plt.figure(figsize=(12,5))
dendrogram = dendrogram(linkage(X,method="ward"))
from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(n_clusters=3,linkage="average",affinity="euclidean")
clusterH = ac.fit_predict(X)
testClassH = [1 if i =="Iris-setosa" else  0 if i =="Iris-versicolor" else 2  for i in data.Species]

count = 0
for i in range(len(clusterH)):
    if clusterH[i] == testClassH[i]:
        count+=1
    
score = count/len(clusterH)
print("Hierarchical Algorithm score: {}".format(score))
plt.figure(figsize=(15,15))

plt.subplot(221)
plt.scatter(data.SepalLengthCm[data.Species == 'Iris-versicolor'], data.PetalLengthCm[data.Species == 'Iris-versicolor'], s = 100, c = 'red', label = 'Iris-versicolor')
plt.scatter(data.SepalLengthCm[data.Species == 'Iris-setosa'], data.PetalLengthCm[data.Species == 'Iris-setosa'], s = 100, c = 'blue', label = 'Iris-setosa')
plt.scatter(data.SepalLengthCm[data.Species == 'Iris-virginica'], data.PetalLengthCm[data.Species == 'Iris-virginica'], s = 100, c = 'green', label = 'Iris-virginica')
plt.legend()
plt.title("Original Data")
plt.xlabel("SepalLengthCm")
plt.ylabel("PetalLengthCm")

plt.subplot(222)
X_kmeans = X.copy()
X_kmeans["Class"] = cluster

plt.scatter(X_kmeans.SepalLengthCm[X_kmeans.Class == 0], X_kmeans.PetalLengthCm[X_kmeans.Class == 0], s = 100, c = 'red', label = 'Iris-versicolor')
plt.scatter(X_kmeans.SepalLengthCm[X_kmeans.Class == 1], X_kmeans.PetalLengthCm[X_kmeans.Class == 1], s = 100, c = 'blue', label = 'Iris-setosa')
plt.scatter(X_kmeans.SepalLengthCm[X_kmeans.Class == 2], X_kmeans.PetalLengthCm[X_kmeans.Class == 2], s = 100, c = 'green', label = 'Iris-virginica')
plt.legend()
plt.title("K-Means")
plt.xlabel("SepalLengthCm")
plt.ylabel("PetalLengthCm")

plt.subplot(223)
X_hier = X.copy()
X_hier["Class"] = clusterH

plt.scatter(X_hier.SepalLengthCm[X_hier.Class == 0], X_hier.PetalLengthCm[X_hier.Class == 0], s = 100, c = 'red', label = 'Iris-versicolor')
plt.scatter(X_hier.SepalLengthCm[X_hier.Class == 1], X_hier.PetalLengthCm[X_hier.Class == 1], s = 100, c = 'blue', label = 'Iris-setosa')
plt.scatter(X_hier.SepalLengthCm[X_hier.Class == 2], X_hier.PetalLengthCm[X_hier.Class == 2], s = 100, c = 'green', label = 'Iris-virginica')
plt.legend()
plt.title("Hierarchical")
plt.xlabel("SepalLengthCm")
plt.ylabel("PetalLengthCm")

plt.show()