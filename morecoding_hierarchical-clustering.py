import numpy as np

#import metaplotlib.pyplot as plt

from sklearn import cluster

from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics import pairwise_distances

from scipy.cluster.hierarchy import dendrogram
def plot_dendrogram(model, **kwargs):

    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node

    counts = np.zeros(model.children_.shape[0])

    n_samples = len(model.labels_)

    for i, merge in enumerate(model.children_):

        current_count = 0

        for child_idx in merge:

            if child_idx < n_samples:

                current_count += 1  # leaf node

            else:

                current_count += counts[child_idx - n_samples]

        counts[i] = current_count



    linkage_matrix = np.column_stack([model.children_, model.distances_,

                                      counts]).astype(float)



    # Plot the corresponding dendrogram

    dendrogram(linkage_matrix, **kwargs)
X = np.array([[1, 2], [1, 4], [1, 0],[4, 2], [4, 4], [4, 0]])

# setting distance_threshold=0 ensures we compute the full tree.

# euclidean is the default metric

# linkage{“ward”, “complete”, “average”, “single”}, default=”ward”

clustering = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage='ward', affinity='euclidean')

clustering.fit(X)

print(clustering.labels_)

plot_dendrogram(clustering, truncate_mode='level', p=3)
clustering2 = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage='complete', affinity='euclidean')

clustering2.fit(X)

print(clustering2.labels_)

plot_dendrogram(clustering2, truncate_mode='level', p=3)


X2 = pairwise_distances(X, metric="euclidean")

print(X2)

#precomputed matrix works for “complete”, “average”, and “single”, but not "ward"

#ward works in feature space

clustering2 = AgglomerativeClustering(distance_threshold=0, n_clusters=None, affinity='precomputed', linkage="complete")

clustering2.fit(X2)

print(clustering2.labels_)

plot_dendrogram(clustering2, truncate_mode='level', p=3)