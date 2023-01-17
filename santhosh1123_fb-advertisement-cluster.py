import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("/kaggle/input/clicks-conversion-tracking/KAG_conversion_data.csv")
data.head()
data.isnull().sum()
data.shape
data.dtypes
data["age"].value_counts()
sns.pairplot(data =data)
data["Total_Conversion"].value_counts()
data["Approved_Conversion"].value_counts()
plt.figure(figsize = ((30,10)))
sns.scatterplot(x = "Total_Conversion", y = "Approved_Conversion" , data = data )
data.head()
x = data.iloc[:,9:11].values
x
from sklearn.cluster import KMeans
wcss = []
for i in range (1,11):
    kmeans = KMeans(n_clusters = i , init = "k-means++" , max_iter = 300, n_init = 10 )
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    # inertia bcoz of wcss 
plt.plot(range(1,11),wcss)
kmeans = KMeans(n_clusters = 3)
y_kmeans = kmeans.fit_predict(x)
y_kmeans
plt.figure(figsize=((20,10)))
plt.scatter (x = x[y_kmeans == 0,0], y = x[y_kmeans == 0,1], c = 'green', label = 'cluster_0' )
plt.scatter (x = x[y_kmeans == 1,0], y = x[y_kmeans == 1,1], c = 'blue', label = 'cluster_1' )
plt.scatter (x = x[y_kmeans == 2,0], y = x[y_kmeans == 2,1], c = 'pink', label = 'cluster_2' )
plt.legend()
plt.xlabel("Total_Conversion")
plt.ylabel("Approved_Conversion")
from sklearn import*
metrics.silhouette_score(x, y_kmeans, metric='euclidean')
plt.figure ( figsize =((10,10)))
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x,method = 'ward'))

from sklearn.cluster import AgglomerativeClustering
agg = AgglomerativeClustering(n_clusters = 3 , affinity = "euclidean")
y_hc = agg.fit_predict(x)
plt.figure(figsize=((20,10)))
plt.scatter (x = x[y_hc == 0,0], y = x[y_hc == 0,1], c = 'green', label = 'cluster_0')
plt.scatter (x = x[y_hc == 1,0], y = x[y_hc == 1,1], c = 'blue', label = 'cluster_1' )
plt.scatter (x = x[y_hc == 2,0], y = x[y_hc == 2,1], c = 'violet', label = 'cluster_2' )
plt.legend()
plt.xlabel("Total_Conversion")
plt.ylabel("Approved_Conversion")
from sklearn import*
metrics.silhouette_score(x, y_hc, metric='euclidean')
from sklearn.metrics import davies_bouldin_score
metrics.davies_bouldin_score(x, y_hc)
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(x)
distances, indices = nbrs.kneighbors(x)
plt.figure(figsize = (10,10))
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples = 5 )
clusters = dbscan.fit_predict(X_scaled)
clusters
plt.figure(figsize=((20,10)))
plt.scatter(x[:, 0], x[:, 1] ,c = clusters)

plt.xlabel("Total_Conversion")
plt.ylabel("Approved_Conversion")
from sklearn.metrics import davies_bouldin_score
metrics.davies_bouldin_score(x, clusters)


plt.figure(figsize=((20,10)))
plt.scatter (x = x[y_hc == 0,0], y = x[y_hc == 0,1], c = 'green', label = 'cluster_0')
plt.scatter (x = x[y_hc == 1,0], y = x[y_hc == 1,1], c = 'blue', label = 'cluster_1' )
plt.scatter (x = x[y_hc == 2,0], y = x[y_hc == 2,1], c = 'violet', label = 'cluster_2' )
plt.legend()
plt.xlabel("Total_Conversion")
plt.ylabel("Approved_Conversion")
plt.title("hierarchical clustering")

cluster_1 = data[y_hc == 0]
cluster_1.head()
cluster_2 = data[y_hc == 1]
cluster_2.head()
cluster_3 = data[y_hc == 2]
cluster_3.head()



data.head()
import seaborn as sns
data.describe()
sns.pairplot(data =data)
plt.figure(figsize = ((30,10)))
sns.scatterplot(y = "Clicks", x = "Impressions" , data = data )
xx = data[["Clicks","Impressions"]].values
xx
plt.figure ( figsize =((30,10)))
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(xx,method = 'ward'))

from sklearn.cluster import AgglomerativeClustering
agg = AgglomerativeClustering(n_clusters = 3 , affinity = "euclidean")
y_hc_xx = agg.fit_predict(xx)
plt.figure(figsize=((20,10)))
plt.scatter (x = x[y_hc_xx == 0,0], y = x[y_hc_xx == 0,1], c = 'green', label = 'cluster_0')
plt.scatter (x = x[y_hc_xx == 1,0], y = x[y_hc_xx == 1,1], c = 'blue', label = 'cluster_1' )
plt.scatter (x = x[y_hc_xx == 2,0], y = x[y_hc_xx == 2,1], c = 'violet', label = 'cluster_2' )
plt.legend()
plt.xlabel("Impressions")
plt.ylabel("Clicks")
from sklearn.cluster import KMeans
wcss = []
for i in range (1,11):
    kmeans = KMeans(n_clusters = i , init = "k-means++" , max_iter = 300, n_init = 10 )
    kmeans.fit(xx)
    wcss.append(kmeans.inertia_)
    # inertia bcoz of wcss 
plt.plot(range(1,11),wcss)
kmeans = KMeans(n_clusters = 3)
y_kmeans_xx = kmeans.fit_predict(xx)
plt.figure(figsize=((20,10)))
plt.scatter (x = x[y_kmeans_xx == 0,0], y = x[y_kmeans_xx == 0,1], c = 'green', label = 'cluster_0' )
plt.scatter (x = x[y_kmeans_xx == 1,0], y = x[y_kmeans_xx == 1,1], c = 'blue', label = 'cluster_1' )
plt.scatter (x = x[y_kmeans_xx == 2,0], y = x[y_kmeans_xx == 2,1], c = 'pink', label = 'cluster_2' )
plt.legend()
plt.xlabel("Impressions")
plt.ylabel("Clicks")
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(xx)
distances, indices = nbrs.kneighbors(xx)
plt.figure(figsize = (10,10))
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled_xx = scaler.fit_transform(xx)


from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples = 5 )
clusters = dbscan.fit_predict(X_scaled_xx)
plt.figure(figsize=((20,10)))
plt.scatter(x[:, 0], x[:, 1] ,c = clusters)
plt.xlabel("Impressions")
plt.ylabel("Clicks")

