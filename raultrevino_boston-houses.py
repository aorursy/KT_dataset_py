import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from scipy.spatial.distance import cdist
data = pd.read_csv("../input/Boston.csv")

data.head()
data.describe()
print("Numero de registros:"+str(data.shape[0]))

for column in data.columns.values:

    print(column + "-NAs:"+ str(pd.isnull(data[column]).values.ravel().sum()))
print("Correlaciones en el dataset:")

data.corr()
plt.matshow(data.corr())
df_norm = (data-data.min())/(data.max() - data.min())

df_norm.head()
from sklearn.cluster import AgglomerativeClustering

clus = AgglomerativeClustering(n_clusters=3,linkage="ward").fit(df_norm)

md_h = pd.Series(clus.labels_)
plt.hist(md_h)

plt.title("Histograma de los clusters")

plt.xlabel("Cluster")

plt.ylabel("Número de casas del cluster")
from scipy.cluster.hierarchy import dendrogram, linkage
Z = linkage(df_norm,"ward")
plt.figure(figsize=(25,10))

plt.title("Dendrograma de las casas")

plt.xlabel("ID de la casa")

plt.ylabel("Distancia")

dendrogram(Z,leaf_rotation=90.,leaf_font_size=8.)

plt.show()
data['cluster_j'] = md_h
data.groupby("cluster_j").mean()
data.groupby("cluster_j").mean().sort_values(['crim'])
max_k = 10## maximo número de clusters que vamos a crear

K = range(1,max_k)

ssw = []



for k in K:

    kmeanModel = KMeans(n_clusters=k).fit(df_norm)

    

    centers = pd.DataFrame(kmeanModel.cluster_centers_)

    labels = kmeanModel.labels_

    

    ssw_k = sum(np.min(cdist(df_norm, kmeanModel.cluster_centers_, "euclidean"), axis = 1))

    ssw.append(ssw_k)

    

#Representación del codo

plt.plot(K, ssw, "bx-")

plt.xlabel("k")

plt.ylabel("SSw(k)")

plt.title("La técnica del codo para encontrar el k óptimo")

plt.show()
model = KMeans(n_clusters=3)

model.fit(df_norm)

md_k = pd.Series(model.labels_)

data["cluster_k"] = md_k
data.head()
data.groupby("cluster_k").mean()
data.groupby("cluster_k").mean().sort_values(['crim'])
data.groupby("cluster_j").mean().sort_values(['crim'])
data.groupby("cluster_k").mean().sort_values(['crim'])