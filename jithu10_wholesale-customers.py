import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#Clustering Libraries
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.cluster import MiniBatchKMeans

from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as shc
from scipy.cluster.hierarchy import dendrogram
data=pd.read_csv('../input/wholesale-customers-data-set/Wholesale customers data.csv')
data.drop(labels=(['Channel','Region']),axis=1,inplace=True)
data.head()

# Basic data Analysis
data.info()
data.describe()
#Standardisation and decomposition

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

PCA_train = PCA(2).fit_transform(scaled_data)

print(scaled_data)



ps = pd.DataFrame(PCA_train)
ps.head()
#Elbow Method
wcss = []
for i in range(1, 25):
    km = KMeans(n_clusters = i, init = 'k-means++', 
                max_iter = 300, n_init = 10, random_state = 0)
    km.fit(ps)
    wcss.append(km.inertia_)
plt.plot(range(1, 25), wcss)
plt.title('The Elbow Method', fontsize = 20)
plt.xlabel('No. of Clusters')
plt.ylabel('wcss')
plt.figure(figsize = (10,5))
plt.show()
#Silhouette Method
no_of_clusters = [4, 5, 6,7] 
print("Average Silhouette Method\n")
for n_clusters in no_of_clusters: 
    cluster = KMeans(n_clusters = n_clusters) 
    cluster_labels = cluster.fit_predict(ps) 
    silhouette_avg = silhouette_score(ps, cluster_labels)
    print("For no of clusters =", n_clusters, 
          "The average silhouette_score is :", silhouette_avg) 

#K Means Clustering
kmean = KMeans(n_clusters=5, random_state=0).fit(ps)
y_kmeans = kmean.predict(ps)
lab = kmean.labels_

plt.figure(figsize=(10,5))
plt.title("KMeans Clustering ",fontsize=20)
plt.scatter(ps[0], ps[1],c = y_kmeans, s=80, 
            cmap='brg',alpha=0.6,marker='D')
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.show()
#Agglomerative Clustering

agc = AgglomerativeClustering(n_clusters=5,affinity = 'euclidean',linkage = 'ward')
y_agc_pred = agc.fit_predict(ps)
plt.figure(figsize =(10,5))
plt.scatter(ps[0], ps[1],c = y_agc_pred, s=80, cmap='brg',alpha=0.6,marker='D')
plt.title('Agglomerative Clustering',fontsize = 20)
plt.show()

plt.figure(figsize=(10,5))
plt.title('Agglomerative Clustering : Dendrogram',fontsize = 20)
dend=shc.dendrogram(shc.linkage(ps,method='ward') ,truncate_mode='level', p=4) 
plt.show()

cluster=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
cluster.fit_predict(ps)
#birch clustering
brc = Birch(branching_factor=500, n_clusters=5, threshold=1.5)
brc.fit(ps)
labels = brc.predict(ps)

plt.title('Birch Clustering',fontsize = 20)
plt.scatter(ps[0], ps[1], c=labels, cmap='brg',alpha=0.6,marker='D')

mb = MiniBatchKMeans(n_clusters=5, random_state=0)
mb.fit(ps)

labels = mb.predict(ps)
plt.title('MiniBatchKMeans clustering',fontsize = 20)
plt.scatter(ps[0], ps[1], c=labels, cmap='brg',alpha=0.6,marker='D')
plt.figure(figsize=(20,15))
plt.show()


