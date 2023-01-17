import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from numpy import unique

from sklearn.cluster import KMeans

from sklearn.cluster import AffinityPropagation

from sklearn.cluster import Birch

from sklearn.cluster import DBSCAN

from sklearn.cluster import MiniBatchKMeans

from sklearn.cluster import MeanShift

from sklearn.cluster import OPTICS

from sklearn.cluster import SpectralClustering

from sklearn.mixture import GaussianMixture

from sklearn.cluster import AgglomerativeClustering

from sklearn import metrics

#                                 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname,filename))
data = pd.read_csv('/kaggle/input/mall-customers/Mall_Customers.csv', index_col=0)

data.head()
data.drop(['Genre'], axis=1, inplace=True)

data.drop(['Age'], axis=1, inplace=True)



data.head()
data = data.sample(frac=1)
data.head()
k_means = KMeans(n_clusters=2)

k_means.fit(data)
k_means.labels_
np.unique(k_means.labels_)
centers = k_means.cluster_centers_



centers
plt.figure(figsize=(10, 8))



plt.scatter(data['Annual Income (k$)'], 

            data['Spending Score (1-100)'], 

            c=k_means.labels_, s=100)



plt.scatter(centers[:,0], centers[:,1], color='blue', marker='s', s=200) 



plt.xlabel('Annual Income')

plt.ylabel('Spending Score')

plt.title('K-Means with 2 clusters')



plt.show()
from sklearn.metrics import silhouette_score



score = silhouette_score (data, k_means.labels_)



print("Score = ", score)
wscc = []

for i in range(1,15): 

    kmeans = KMeans(n_clusters=i, init="k-means++",random_state=0)

    kmeans.fit(data)

    wscc.append(kmeans.inertia_)  



plt.plot(range(1,15),wscc,marker="*",c="black")

plt.title("Elbow plot for optimal number of clusters")
k_means = KMeans(n_clusters=5)

k_means.fit(data)
np.unique(k_means.labels_)
centers = k_means.cluster_centers_



centers
plt.figure(figsize=(10, 8))



plt.scatter(data['Annual Income (k$)'], 

            data['Spending Score (1-100)'], 

            c=k_means.labels_, s=100)



plt.scatter(centers[:,0], centers[:,1], color='blue', marker='s', s=200) 



plt.xlabel('Annual Income')

plt.ylabel('Spending Score')

plt.title('5 Cluster K-Means')



plt.show()
score = metrics.silhouette_score(data, k_means.labels_)



print("Score = ", score)
score1 = metrics.silhouette_samples(data, k_means.labels_, metric='euclidean')

print("Score = ", score1)
model_aff = AffinityPropagation(damping=0.9)

model_aff.fit(data)

#

yhat_aff = model_aff.predict(data)

clusters_aff = unique(yhat_aff)

print("Clusters of Affinity Prop.",clusters_aff)

labels_aff = model_aff.labels_

centroids_aff = model_aff.cluster_centers_
plt.figure(figsize=(10, 8))



plt.scatter(data['Annual Income (k$)'], 

            data['Spending Score (1-100)'], 

            c=labels_aff, s=100)



plt.scatter(centroids_aff[:,0], centroids_aff[:,1], color='red', marker='*', s=200) 



plt.xlabel('Annual Income')

plt.ylabel('Spending Score')

plt.title('Affinity Propagation')

plt.grid()

plt.show()
score_aff = metrics.silhouette_score(data,labels_aff)



print("Score of Affinity Propagation = ", score_aff)
model_br = Birch(threshold=0.01, n_clusters=5)

model_br.fit(data)

#

yhat_br = model_br.predict(data)

clusters_br = unique(yhat_br)

print("Clusters of Birch",clusters_br)

labels_br = model_br.labels_
score_br = metrics.silhouette_score(data,labels_br)



print("Score of Birch = ", score_br)
# dbscan clustering

from numpy import unique

from numpy import where

data_X = data.iloc[:,[0,1]].values
# define the model

model = DBSCAN(eps=0.7, min_samples=90)

# fit model and predict clusters

yhat = model.fit_predict(data_X)

# retrieve unique clusters

clusters = unique(yhat)

# create scatter plot for samples from each cluster

for cluster in clusters:

	# get row indexes for samples with this cluster

	row_ix = where(yhat == cluster)

	# create scatter of these samples

	plt.scatter(data_X[row_ix, 0], data_X[row_ix, 1])

# show the plot

plt.show()
model_mini = MiniBatchKMeans(n_clusters=2)

model_mini.fit(data)

#

yhat_mini = model_mini.predict(data)

clusters_mini = unique(yhat_mini)

print("Clusters of Mini Batch KMeans.",clusters_mini)

labels_mini = model_mini.labels_

centroids_mini = model_mini.cluster_centers_
wscc = []

for i in range(1,15): 

    mkmeans = MiniBatchKMeans(n_clusters=i, init="k-means++",random_state=0)

    mkmeans.fit(data)

    wscc.append(mkmeans.inertia_)  



plt.plot(range(1,15),wscc,marker="*",c="black")

plt.title("Elbow plot for Mini Batch KMeans")
model_mini = MiniBatchKMeans(n_clusters=5)

model_mini.fit(data)

#

yhat_mini = model_mini.predict(data)

clusters_mini = unique(yhat_mini)

print("Clusters of Mini Batch KMeans.",clusters_mini)

labels_mini = model_mini.labels_

centroids_mini = model_mini.cluster_centers_
plt.figure(figsize=(10, 8))



plt.scatter(data['Annual Income (k$)'], 

            data['Spending Score (1-100)'], 

            c=labels_mini, s=100)



plt.scatter(centroids_mini[:,0], centroids_mini[:,1], color='red', marker='*', s=200) 



plt.xlabel('Annual Income')

plt.ylabel('Spending Score')

plt.title('Mini Batch KMeans')

plt.grid()

plt.show()
score_mini = metrics.silhouette_score(data,labels_mini)



print("Score of Birch = ", score_mini)
model_ms = MeanShift(bandwidth=25)

model_ms.fit(data)

#

yhat_ms = model_ms.predict(data)

clusters_ms = unique(yhat_ms)

print("Clusters of Mean Shift.",clusters_ms)

labels_ms = model_ms.labels_

centroids_ms = model_ms.cluster_centers_
plt.figure(figsize=(10, 8))



plt.scatter(data['Annual Income (k$)'], 

            data['Spending Score (1-100)'], 

            c=labels_ms, s=100)



plt.scatter(centroids_ms[:,0], centroids_ms[:,1], color='red', marker='*', s=200) 



plt.xlabel('Annual Income')

plt.ylabel('Spending Score')

plt.title('Mean Shift')

plt.grid()

plt.show()
score_ms = metrics.silhouette_score(data,labels_ms)



print("Score of Mean Shift = ", score_ms)
model_op = OPTICS(eps=0.8, min_samples=10)

#

yhat_op = model_op.fit_predict(data)

clusters_op = unique(yhat_op)

print("Clusters of Mean Shift.",clusters_op)

labels_op = model_op.labels_
score_op = metrics.silhouette_score(data,labels_op)



print("Score of Mean Shift = ", score_op)
model_sc = SpectralClustering(n_clusters=5)

#

yhat_sc = model_sc.fit_predict(data)

clusters_sc = unique(yhat_sc)

print("Clusters of Mean Shift.",clusters_sc)

labels_sc = model_sc.labels_
score_sc = metrics.silhouette_score(data,labels_sc)



print("Score of Mean Shift = ", score_sc)
from numpy import unique

from numpy import where

data_X = data.iloc[:,[0,1]].values
model_gb = GaussianMixture(n_components=5)

model_gb.fit(data_X)

#

yhat_gb = model_gb.predict(data_X)

clusters_gb = unique(yhat_gb)

# create scatter plot for samples from each cluster

for cluster in clusters_gb:

	# get row indexes for samples with this cluster

	row_ix = where(yhat_gb == cluster)

	# create scatter of these samples

	plt.scatter(data_X[row_ix, 0], data_X[row_ix, 1])

# show the plot

plt.show()
model_agg = AgglomerativeClustering(n_clusters=5)

#

yhat_agg = model_agg.fit_predict(data)

clusters_agg = unique(yhat_agg)

print("Clusters of Mini Batch KMeans.",clusters_agg)

labels_agg = model_agg.labels_
score_agg = metrics.silhouette_score(data,labels_agg)



print("Score of Mean Shift = ", score_agg)