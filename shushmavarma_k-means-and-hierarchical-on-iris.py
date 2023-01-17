import numpy as np

import pandas as pd

from sklearn import datasets
iris = datasets.load_iris()
X_iris = iris.data

y_iris = iris.target
from sklearn.preprocessing import scale

X_scaled = pd.DataFrame(scale(X_iris))
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt



Ks = [2,3,4,5,6,7,8,9]

ssw=[]

for k in Ks:

    kmeans=KMeans(n_clusters=int(k))

    kmeans.fit(X_scaled)

    sil_score=silhouette_score(X_scaled,kmeans.labels_)

    print("silhouette score:",sil_score,"number of clusters are:", int(k))

    ssw.append(kmeans.inertia_)

plt.plot(Ks,ssw)
k = 3

kmeans = KMeans(n_clusters=k)

kmeans.fit(X_scaled)
labels1 = kmeans.labels_

X_scaled["cluster"]=labels1
for i in range(k):

    ds = X_scaled[X_scaled["cluster"]==i].as_matrix()

    plt.plot(ds[:,0],ds[:,1],'o')



plt.show()
kmeans.inertia_
from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.cluster import AgglomerativeClustering
for n_clusters in range(2,10):

    cluster_model = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean',linkage='single')

    cluster_labels = cluster_model.fit_predict(X_scaled)

    silhouette_avg = silhouette_score(X_scaled,cluster_labels,metric='euclidean')

    print("For n_clusters =", n_clusters, 

          "The average silhouette_score is:", silhouette_avg)
for n_clusters in range(2,10):

    cluster_model = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean',linkage='ward')

    cluster_labels = cluster_model.fit_predict(X_scaled)

    silhouette_avg = silhouette_score(X_scaled,cluster_labels,metric='euclidean')

    print("For n_clusters =", n_clusters, 

          "The average silhouette_score is:", silhouette_avg)
s = 3

hclust = AgglomerativeClustering(n_clusters=s, affinity='euclidean',linkage='single')

hclust.fit(X_scaled)
hclust1 = AgglomerativeClustering(n_clusters=s, affinity='euclidean',linkage='complete')

hclust1.fit(X_scaled)
hclust2 = AgglomerativeClustering(n_clusters=s, affinity='euclidean',linkage='average')

hclust2.fit(X_scaled)
hclust3 = AgglomerativeClustering(n_clusters=s, affinity='euclidean',linkage='ward')

hclust3.fit(X_scaled)
labels = hclust.fit_predict(X_scaled)

X_scaled["cluster"]=labels
for i in range(s):

    hc = X_scaled[X_scaled["cluster"]==i].as_matrix()

    plt.plot(hc[:,0],hc[:,1],'o')

plt.show()
# SINGLE
Z = linkage(X_scaled, 'single')

plt.figure(figsize=(10, 10))

plt.title('Agglomerative Hierarchical Clustering Dendogram')

plt.xlabel('Cluster points')

plt.ylabel('Distance')

dendrogram(Z, leaf_rotation=90.,color_threshold = 40, leaf_font_size=8. )

plt.tight_layout()
# COMPLETE



Z1 = linkage(X_scaled, 'complete')

plt.figure(figsize=(10, 10))

plt.title('Agglomerative Hierarchical Clustering Dendogram')

plt.xlabel('Cluster points')

plt.ylabel('Distance')

dendrogram(Z1, leaf_rotation=90.,color_threshold = 40, leaf_font_size=8. )

plt.tight_layout()
# AVERAGE



Z2 = linkage(X_scaled, 'average')

plt.figure(figsize=(10, 10))

plt.title('Agglomerative Hierarchical Clustering Dendogram')

plt.xlabel('Cluster points')

plt.ylabel('Distance')

dendrogram(Z2, leaf_rotation=90.,color_threshold = 40, leaf_font_size=8. )

plt.tight_layout()
# WARD



Z33 = linkage(X_scaled, 'ward')

plt.figure(figsize=(10, 10))

plt.title('Agglomerative Hierarchical Clustering Dendogram')

plt.xlabel('Cluster points')

plt.ylabel('Distance')

dendrogram(Z, leaf_rotation=90.,color_threshold = 40, leaf_font_size=8. )

plt.tight_layout()