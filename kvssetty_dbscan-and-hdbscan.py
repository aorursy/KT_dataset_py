import matplotlib.pyplot as plt

from sklearn import datasets

%matplotlib inline



#Toy data sets

centers_neat = [(-10, 10), (0, -5), (10, 5)]

x_neat, _ = datasets.make_blobs(n_samples=5000,

                                centers=centers_neat,

                                cluster_std=2,

                                random_state=2)



x_clumsy, _ = datasets.make_classification(n_samples=5000,

                                          n_features=10,

                                          n_classes=3,

                                          n_clusters_per_class=1,

                                          class_sep=1.5,

                                          shuffle=False,

                                          random_state=301)

#Default plot params

plt.style.use('seaborn')

cmap = 'tab10'



plt.figure(figsize=(17,8))

plt.subplot(121, title='"Neat" Clusters')

plt.scatter(x_neat[:,0], x_neat[:,1])

plt.subplot(122, title='"Clumsy" Clusters')

plt.scatter(x_clumsy[:,0], x_clumsy[:,1])
from sklearn.cluster import KMeans



#Predict K-Means cluster membership

km_neat_learn = KMeans(n_clusters=3, random_state=2).fit(x_neat)

km_neat = KMeans(n_clusters=3, random_state=2).fit_predict(x_neat)



km_clumsy_learn = KMeans(n_clusters=3, random_state=2).fit(x_clumsy)

km_clumsy = KMeans(n_clusters=3, random_state=2).fit_predict(x_clumsy)



plt.figure(figsize=(15,8))

plt.subplot(121, title='"Neat" K-Means')

plt.scatter(x_neat[:,0], x_neat[:,1], c=km_neat, cmap=cmap)

plt.scatter(km_neat_learn.cluster_centers_[:,0], km_neat_learn.cluster_centers_[:,1],

            marker='X', s=150, c='black')





plt.subplot(122, title='"Clumsy" K-Means')

plt.scatter(x_clumsy[:,0], x_clumsy[:,1], c=km_clumsy, cmap=cmap)

plt.scatter(km_clumsy_learn.cluster_centers_[:,0], km_clumsy_learn.cluster_centers_[:,1],

            marker='X', s=150, c='black')
# let GMM code and comments come here.
from sklearn.cluster import DBSCAN



#Predict DBSCAN cluster membership

dbscan_neat = DBSCAN().fit_predict(x_neat)

dbscan_clumsy = DBSCAN().fit_predict(x_clumsy)



plt.figure(figsize=(15,8))

plt.subplot(121, title='"Neat" DBSCAN')

plt.scatter(x_neat[:,0], x_neat[:,1], c=dbscan_neat, cmap=cmap)

plt.subplot(122, title='"Clumsy" DBSCAN')

plt.scatter(x_clumsy[:,0], x_clumsy[:,1], c=dbscan_clumsy, cmap=cmap)
print("DBSCAN memberships of dbscan_neat:\n{}".format(dbscan_neat))
print("DBSCAN memberships of dbscan_clumsy:\n{}".format(dbscan_clumsy))




#Predict DBSCAN cluster membership



dbscan = DBSCAN(eps=1.62, min_samples=4).fit(x_clumsy)

dbscan_clumsy01 = dbscan.fit_predict(x_clumsy)



plt.figure(figsize=(17,8))





plt.subplot(122, title='"Clumsy" DBSCAN')

plt.scatter(x_clumsy[:,0], x_clumsy[:,1], c=dbscan_clumsy01, cmap=cmap)
print("DBSCAN memberships of dbscan_clumsy01:\n{}".format(dbscan_clumsy01))
import numpy as np

np.unique(dbscan_clumsy01)
core_sample_indicies_array = dbscan.core_sample_indices_

core_sample_indicies_array.shape
import hdbscan
clust_count = np.linspace(1, 20, num=20, dtype='int')



clust_number = 2

plot_number = 1

plt.figure (figsize=(17,12))

while clust_number < 16:

    hdb = hdbscan.HDBSCAN(min_cluster_size=clust_number)

    hdb_pred = hdb.fit(x_clumsy)

    plt.subplot(5, 4, plot_number, title = 'Min. Cluster Size = {}'.format(clust_number))

    plt.scatter(x_clumsy[:,0], x_clumsy[:,1], c=hdb_pred.labels_, cmap=cmap)

    plot_number += 1

    clust_number += 1



plt.tight_layout()
np.unique(hdb_pred.labels_)