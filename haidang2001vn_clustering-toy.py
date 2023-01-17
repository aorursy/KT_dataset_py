import numpy as np

import matplotlib

import matplotlib.pyplot as plt

from sklearn.datasets import make_moons, make_blobs

from sklearn.cluster import KMeans

from sklearn.cluster import DBSCAN
# Define datasets

n_samples = 300

blobs_params = dict(random_state=0, n_features=2)

X1,labels_true_1 = make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=[0.5, 0.5],**blobs_params)

# make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=[1.5, .3], **blobs_params)

  

X2, labels_true_2 = make_moons(n_samples=n_samples, noise=.05, random_state=0)

#                           - np.array([0.5, 0.25]))

centers = [[1, 1], [-1, -1], [1, -1]]

n_clusters = len(centers)

X3, labels_true_3 = make_blobs(n_samples=3000, centers=centers, cluster_std=0.7,random_state=0)
plt.subplot(1, 3, 1)

plt.scatter(X1[:, 0], X1[:, 1], s=10, color='blue')

plt.xlim(-5,5)

plt.ylim(-5, 5)

plt.title("Data X1")



plt.subplot(1, 3, 2)

plt.scatter(X2[:, 0], X2[:, 1], s=10, color='red')

plt.xlim(-5,5)

plt.ylim(-5, 5)

plt.title("Data X2")



plt.subplot(1, 3, 3)

plt.scatter(X3[:, 0], X3[:, 1], s=10, color='green')

plt.xlim(-5,5)

plt.ylim(-5, 5)

plt.title("Data X3")
# Compute clustering with Means

# for X1

k_means_X1 = KMeans(init='k-means++', n_clusters=2, n_init=10)

k_means_X1.fit(X1)



### for X2

k_means_X2 = KMeans(init='k-means++', n_clusters=2, n_init=10, verbose=1)

k_means_X2.fit(X2)



# for X3 

k_means_X3 = KMeans(init='k-means++', n_clusters=5, n_init=10)

k_means_X3.fit(X3)
from sklearn.metrics.pairwise import pairwise_distances_argmin

fig = plt.figure(figsize=(8, 3))

fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)



colors = ['yellow', 'cyan']

#### for data X1

k_means = k_means_X1

k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)

k_means_labels = pairwise_distances_argmin(X1, k_means_cluster_centers)

# KMeans

ax = fig.add_subplot(1, 1, 1)

for k, col in zip(range(n_clusters), colors):

    my_members = k_means_labels == k

    cluster_center = k_means_cluster_centers[k]

    ax.plot(X1[my_members, 0], X1[my_members, 1], 'w',

            markerfacecolor=col, marker='.',markersize=10)

    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,

            markeredgecolor='k', markersize=16)

ax.set_title('KMeans for data X1')

ax.set_xticks(())

ax.set_yticks(())
from sklearn.metrics.pairwise import pairwise_distances_argmin

fig = plt.figure(figsize=(8, 3))

fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)

colors = ['magenta', 'lime']



#### for data X2

k_means = k_means_X2

k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)

k_means_labels = pairwise_distances_argmin(X2, k_means_cluster_centers)

# KMeans

ax = fig.add_subplot(1, 1, 1)

for k, col in zip(range(n_clusters), colors):

    my_members = k_means_labels == k

    cluster_center = k_means_cluster_centers[k]

    ax.plot(X2[my_members, 0], X2[my_members, 1], 'w',

            markerfacecolor=col, marker='.',markersize=10)

    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,

            markeredgecolor='k', markersize=16)

ax.set_title('KMeans for data X2')

ax.set_xticks(())

ax.set_yticks(())
from sklearn.metrics.pairwise import pairwise_distances_argmin

fig = plt.figure(figsize=(8, 3))

fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)

colors = ['red', 'green', 'blue']



#### for data X2

k_means = k_means_X3

k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)

k_means_labels = pairwise_distances_argmin(X3, k_means_cluster_centers)

# KMeans

ax = fig.add_subplot(1, 1, 1)

for k, col in zip(range(n_clusters), colors):

    my_members = k_means_labels == k

    cluster_center = k_means_cluster_centers[k]

    ax.plot(X3[my_members, 0], X3[my_members, 1], 'w',

            markerfacecolor=col, marker='.',markersize=10)

    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,

            markeredgecolor='k', markersize=16)

ax.set_title('KMeans for data X3')

#############3 TEST DBSCAN

# Compute DBSCAN for data X1

dbscan_X1 = DBSCAN(eps=3, min_samples=5).fit(X1)

core_samples_mask_X1 = np.zeros_like(dbscan_X1.labels_, dtype=bool)

core_samples_mask_X1[dbscan_X1.core_sample_indices_] = True

labels_X1 = dbscan_X1.labels_

# Number of clusters in labels, ignoring noise if present.

n_clusters_X1 = len(set(labels_X1)) - (1 if -1 in labels_X1 else 0)

n_noise_X1 = list(labels_X1).count(-1)

print(dbscan_X1)

print(core_samples_mask_X1)

print("labels_X1",labels_X1)

print("n_clusters_",n_clusters_X1)

print("n_noise_X1",n_noise_X1)



# Black removed and is used for noise instead.

labels = labels_X1

X = X1

core_samples_mask = core_samples_mask_X1

n_clusters_ = n_clusters_X1



unique_labels = set(labels)

colors = [plt.cm.Spectral(each)

          for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):

    if k == -1:

        # Black used for noise.

        col = [0, 0, 0, 1]



    class_member_mask = (labels == k)



    xy = X[class_member_mask & core_samples_mask]

    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),

             markeredgecolor='k', markersize=14)



    xy = X[class_member_mask & ~core_samples_mask]

    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),

             markeredgecolor='k', markersize=6)



plt.title(' DBSCAN - Estimated number of clusters for data X1 : %d' % n_clusters_)

plt.show()
#############3 TEST DBSCAN

# Compute DBSCAN for data X2

dbscan_X2 = DBSCAN(eps=0.5, min_samples=5).fit(X2) ## change eps to see what happen

core_samples_mask_X2 = np.zeros_like(dbscan_X2.labels_, dtype=bool)

core_samples_mask_X2[dbscan_X2.core_sample_indices_] = True

labels_X2 = dbscan_X2.labels_

# Number of clusters in labels, ignoring noise if present.

n_clusters_X2 = len(set(labels_X2)) - (1 if -1 in labels_X1 else 0)

n_noise_X2 = list(labels_X2).count(-1)



# Black removed and is used for noise instead.

labels = labels_X2

X = X2

core_samples_mask = core_samples_mask_X2

n_clusters_ = n_clusters_X2



unique_labels = set(labels)

colors = [plt.cm.Spectral(each)

          for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):

    if k == -1:

        # Black used for noise.

        col = [0, 0, 0, 1]



    class_member_mask = (labels == k)



    xy = X[class_member_mask & core_samples_mask]

    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),

             markeredgecolor='k', markersize=14)



    xy = X[class_member_mask & ~core_samples_mask]

    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),

             markeredgecolor='k', markersize=6)



plt.title('DBSCAN - Estimated number of clusters for data X2 : %d' % n_clusters_)

plt.show()
#############3 TEST DBSCAN

# Compute DBSCAN for data X3

dbscan_X3 = DBSCAN(eps=0.3, min_samples=2).fit(X3) ## change eps to see what happen

core_samples_mask_X3 = np.zeros_like(dbscan_X3.labels_, dtype=bool)

core_samples_mask_X3[dbscan_X3.core_sample_indices_] = True

labels_X3 = dbscan_X3.labels_

# Number of clusters in labels, ignoring noise if present.

n_clusters_X3 = len(set(labels_X3)) - (1 if -1 in labels_X3 else 0)

n_noise_X3 = list(labels_X3).count(-1)



# Black removed and is used for noise instead.

labels = labels_X3

X = X3

core_samples_mask = core_samples_mask_X3

n_clusters_ = n_clusters_X3



unique_labels = set(labels)

colors = [plt.cm.Spectral(each)

          for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):

    if k == -1:

        # Black used for noise.

        col = [0, 0, 0, 1]



    class_member_mask = (labels == k)



    xy = X[class_member_mask & core_samples_mask]

    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),

             markeredgecolor='k', markersize=14)



    xy = X[class_member_mask & ~core_samples_mask]

    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),

             markeredgecolor='k', markersize=6)



plt.title('DBSCAN - Estimated number of clusters for data X3 : %d' % n_clusters_)

plt.show()