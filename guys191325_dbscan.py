def plot(X):# plot

    plt.scatter(X[:, 0], X[:, 1])

    plt.xlabel("Feature 0")

    plt.ylabel("Feature 1")

    plt.show()

    

import numpy as np

import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs

from sklearn.cluster import KMeans

from sklearn import preprocessing
# generate some random cluster data

X, y = make_blobs(random_state=170, n_samples=500, centers = 5)

rng = np.random.RandomState(74)

# transform the data to be stretched

transformation = rng.normal(size=(2, 2))

X = np.dot(X, transformation)



plot(X)
outliers = 30 * (np.random.RandomState(42).rand(100, 2) ) - 15.0

X = np.concatenate([X, outliers])

y = np.concatenate([y, [-1]*len(outliers)])

plot(X)
# cluster the data into five clusters

kmeans = KMeans(n_clusters=5)

kmeans.fit(X)

kmeans_clusters = kmeans.predict(X)

# plot the cluster assignments and cluster centers

plt.scatter(X[:, 0], X[:, 1], c=kmeans_clusters, cmap="plasma")

plt.scatter(kmeans.cluster_centers_[:, 0],   

            kmeans.cluster_centers_[:, 1],

            marker='^', 

            c=[0, 1, 2, 3, 4], 

            s=100, 

            linewidth=2,

            cmap="plasma")

plt.xlabel("Feature 0")

plt.ylabel("Feature 1")

MIN_PTS=50

from sklearn.cluster import DBSCAN

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# cluster the data into five clusters

dbscan = DBSCAN(eps=0.2, min_samples = MIN_PTS)

dbscan_clusters = dbscan.fit_predict(X_scaled)

# plot the cluster assignments

plt.scatter(X[:, 0], X[:, 1], c=dbscan_clusters, cmap="plasma")

plt.xlabel("Feature 0")

plt.ylabel("Feature 1")

print(set(dbscan_clusters))
print('num of classified noise instances:',len([x for x in dbscan.labels_ if x==-1]))

print('num of classified regular instances:',len([x for x in dbscan.labels_ if x!=-1]))

print('Estimated number of clusters: %d' % len(set(dbscan.labels_) - set([-1])))
from sklearn.neighbors import NearestNeighbors

from scipy.signal import savgol_filter



neigh = NearestNeighbors(n_neighbors=MIN_PTS)

nbrs = neigh.fit(X)

distances, indices = nbrs.kneighbors(X)

distances = np.sort(distances, axis=0)

distances = distances[:,1]

distances = savgol_filter(distances, 21, 1) # Smooth the data

plt.plot(distances)

from sklearn.cluster import OPTICS

clustering_optics = OPTICS(min_samples=50).fit(X)

optics_clusters = clustering_optics.labels_

plt.scatter(X[:, 0], X[:, 1], c=optics_clusters, cmap="plasma")

plt.xlabel("Feature 0")

plt.ylabel("Feature 1")

print(set(optics_clusters))


from sklearn.metrics import silhouette_score

from sklearn.metrics.cluster import adjusted_rand_score

#k-means performance:

print("K means ARI = %0.3f" % (adjusted_rand_score(y, kmeans_clusters)))

print("DBSCAN ARI = %0.3f" % (adjusted_rand_score(y, dbscan_clusters)))

print("OPTICS ARI = %0.3f" % (adjusted_rand_score(y, optics_clusters)))

print('K means silhouette_score %0.3f' % (silhouette_score(X,kmeans_clusters)))

print('DBSCAN silhouette_score %0.3f' % (silhouette_score(X,dbscan_clusters)))

print('OPTICS silhouette_score %0.3f' % (silhouette_score(X,optics_clusters)))
