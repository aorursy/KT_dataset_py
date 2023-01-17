import pandas as pd 

import numpy as np

from matplotlib import pyplot as plt



data = pd.read_table("/kaggle/input/data-to-work-on-basic-examples/x4.txt", header= None)
from sklearn.cluster import KMeans

wcss = []



for k  in range(2,7):

    kmeans = KMeans(n_clusters=k)

    kmeans.fit(data)

    wcss.append(kmeans.inertia_)

    

plt.plot(range(2,7),wcss)

plt.xlabel("number of k(cluster) value")

plt.ylabel("wcss(within cluster sum of squares)")

plt.show()
import numpy as np



from matplotlib import pyplot as plt

from scipy.cluster.hierarchy import dendrogram

from sklearn.datasets import load_iris

from sklearn.cluster import AgglomerativeClustering





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





iris = load_iris()

X = iris.data



# setting distance_threshold=0 ensures we compute the full tree.

model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)



model = model.fit(data)

plt.title('Hierarchical Clustering Dendrogram')

# plot the top three levels of the dendrogram

plot_dendrogram(model, truncate_mode='level', p=3)

plt.xlabel("Number of points in node (or index of point if no parenthesis).")

plt.show()

print( )