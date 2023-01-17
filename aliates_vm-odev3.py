from pandas import DataFrame

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

import pandas as pd

import os



print(os.listdir("../input"))
df = pd.read_csv("../input/k-means_clustering_dataset.txt", '\t')

df
kmeans = KMeans(n_clusters=3).fit(df)

centroids = kmeans.cluster_centers_

print(centroids)
plt.scatter(df['x'], df['y'], c= kmeans.labels_.astype(float), s=50, alpha=1)

plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
from sklearn.cluster import AgglomerativeClustering



cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')  

cluster.fit_predict(df)
plt.figure()  

plt.scatter(df.iloc[:,0], df.iloc[:,1], c=cluster.labels_, cmap='rainbow') 