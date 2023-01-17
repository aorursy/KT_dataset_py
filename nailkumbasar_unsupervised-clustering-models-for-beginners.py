import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



d1 = pd.read_csv("../input/wine-quality-clustering-unsupervised/winequality-red.csv") 



d2 = d1.drop(["quality"], axis=1)





dictionary = {"fixed_acidity":d2.fixed_acidity.values,"volatile_acidity":d2.volatile_acidity.values}

d2 = pd.DataFrame(dictionary)

d2.head()
from sklearn.cluster import KMeans

wcss = []



for k in range(1,15):

    kmeans = KMeans(n_clusters=k)

    kmeans.fit(d2)

    wcss.append(kmeans.inertia_)

    

plt.plot(range(1,15),wcss)

plt.title('The Elbow Method')

plt.xlabel("number of k (cluster) value")

plt.ylabel("wcss")

plt.show()
#%% k = 3 icin modelim



kmeans2 = KMeans(n_clusters=3)



clusters = kmeans2.fit_predict(d2)



d2["clusters"] = clusters



d2.groupby("clusters").count()

plt.xlabel("fixed_acidity")

plt.ylabel("volatile_acidity")



plt.scatter(d2.fixed_acidity[d2.clusters == 0 ],d2.volatile_acidity[d2.clusters == 0],color = "red")

plt.scatter(d2.fixed_acidity[d2.clusters == 1 ],d2.volatile_acidity[d2.clusters == 1],color = "green")

plt.scatter(d2.fixed_acidity[d2.clusters == 2 ],d2.volatile_acidity[d2.clusters == 2],color = "blue")



plt.scatter(kmeans2.cluster_centers_[:,0],kmeans2.cluster_centers_[:,1],color = "yellow")

plt.show()
from scipy.cluster.hierarchy import linkage, dendrogram



merg = linkage(d2,method="ward")

dendrogram(merg,leaf_rotation = 90)

plt.xlabel("data points")

plt.ylabel("euclidean distance")

plt.show()
from sklearn.cluster import AgglomerativeClustering



hiyerartical_cluster = AgglomerativeClustering(n_clusters = 5,affinity= "euclidean",linkage = "ward")

cluster = hiyerartical_cluster.fit_predict(d2)



d2["label"] = cluster



plt.xlabel("fixed_acidity")

plt.ylabel("volatile_acidity")



plt.scatter(d2.fixed_acidity[d2.clusters == 0 ],d2.volatile_acidity[d2.clusters == 0],color = "red")

plt.scatter(d2.fixed_acidity[d2.clusters == 1 ],d2.volatile_acidity[d2.clusters == 1],color = "green")

plt.scatter(d2.fixed_acidity[d2.clusters == 2 ],d2.volatile_acidity[d2.clusters == 2],color = "blue")

plt.show()