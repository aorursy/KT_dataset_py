import numpy as np

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import scipy as sp



from sklearn.cluster import KMeans

from warnings import filterwarnings

filterwarnings('ignore')
df = pd.read_csv("../input/test123/USArrests.csv").copy()

df.head()
df.index = df.iloc[:,0]

df.head()
df = df.iloc[:,1:5]

df.head()
df.index.name = None

df.head()
df.info()
df.isnull().sum()
df.describe().T
df.hist(figsize=(10,10));
from sklearn.cluster import KMeans



kmeans = KMeans(n_clusters = 4)

kmeans
k_fit = kmeans.fit(df)

k_fit.n_clusters
k_fit.cluster_centers_
k_fit.labels_
kmeans = KMeans(n_clusters = 2)

k_fit = kmeans.fit(df)
clusters = k_fit.labels_

plt.scatter(df.iloc[:,0], df.iloc[:,1],c = clusters,s = 50, cmap="viridis");

centers = k_fit.cluster_centers_



plt.scatter(centers[:,0], centers[:,1],c="black", s=200, alpha=0.5);
from mpl_toolkits.mplot3d import Axes3D
kmeans = KMeans(n_clusters=3)

k_fit = kmeans.fit(df)

clusters = k_fit.labels_

centers = kmeans.cluster_centers_
plt.rcParams["figure.figsize"] = (16,9)

fig = plt.figure()

ax = Axes3D(fig)

ax.scatter(df.iloc[:,0],df.iloc[:,1],df.iloc[:,2]);
fig = plt.figure()

ax = Axes3D(fig)

ax.scatter(df.iloc[:,0],df.iloc[:,1],df.iloc[:,2],c =clusters);

ax.scatter(centers[:,0],centers[:,1],centers[:,2], marker="*",c="#050505",s=500);
kmeans = KMeans(n_clusters = 3)

k_fit = kmeans.fit(df)

clusters = k_fit.labels_
pd.DataFrame({"States" : df.index, "Clusters": clusters})[:10]
df["Cluster Number"] = clusters

df.head(10)
df["Cluster Number"] = df["Cluster Number"] + 1

df.head(10)
!pip install yellowbrick
from yellowbrick.cluster import KElbowVisualizer



kmeans = KMeans()

visualizer = KElbowVisualizer(kmeans, k=(2,50))

visualizer.fit(df); 

visualizer.poof();  
kmeans = KMeans(n_clusters = 6)

k_fit = kmeans.fit(df)

clusters = k_fit.labels_
pd.DataFrame({"States" : df.index, "Clusters": clusters})[0:10]
df = pd.read_csv("../input/test123/USArrests.csv").copy()

df.index = df.iloc[:,0]

df = df.iloc[:,1:5]



df.index.name = None

df.head()
from scipy.cluster.hierarchy import linkage



hc_complete = linkage(df, "complete")

hc_average = linkage(df, "average")

hc_single = linkage(df, "single")
from scipy.cluster.hierarchy import dendrogram



plt.figure(figsize=(15, 10))

plt.title('Hierarchical Clustering - Dendogram')

plt.xlabel('Indexes')

plt.ylabel('Distance')

dendrogram(

    hc_complete,

    leaf_font_size=10);
from scipy.cluster.hierarchy import dendrogram



plt.figure(figsize=(15, 10))

plt.title('Hierarchical Clustering - Dendogram')

plt.xlabel('Indexes')

plt.ylabel('Distance')

dendrogram(

    hc_complete,

    truncate_mode = "lastp",

    p = 4,

    show_contracted = True);
from scipy.cluster.hierarchy import dendrogram



plt.figure(figsize=(15, 10))

plt.title('Hierarchical Clustering - Dendogram')

plt.xlabel('Indexes')

plt.ylabel('Distance')

den = dendrogram(

    hc_complete,

    leaf_font_size=10);
from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters = 4, 

                                  affinity = "euclidean", 

                                  linkage = "ward")



cluster.fit_predict(df)
pd.DataFrame({"States" : df.index, "Clusters": cluster.fit_predict(df)})[0:10]
df["Cluster_number"] = cluster.fit_predict(df)

df.head()
df = pd.read_csv("../input/test123/USArrests.csv").copy()

df.index = df.iloc[:,0]

df = df.iloc[:,1:5]



df.index.name = None

df.head()
from sklearn.decomposition import PCA



pca = PCA(n_components=3)

pca_fit = pca.fit_transform(df)
component_df = pd.DataFrame(data=pca_fit,columns= ["First Component","Second Component","Third Component"])

component_df.head()
pca.explained_variance_ratio_
pca = PCA().fit(df)
plt.plot(np.cumsum(pca.explained_variance_ratio_));