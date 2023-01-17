import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



from sklearn import manifold #needed for multidimensional scaling (MDS) and t-SNE

from sklearn import cluster #needed for k-Means clustering

from sklearn import preprocessing #needed for scaling attributes to the nterval [0,1]
df = pd.read_csv('../input/iris_nolabels.csv')

data = np.array(df.values, dtype=float)

print('(number of examples, number of attributes): ', data.shape)
min_max_scaler = preprocessing.MinMaxScaler()

data = min_max_scaler.fit_transform(data)
colors = np.array(['orange', 'blue', 'lime', 'blue', 'khaki', 'pink', 'green', 'purple'])



# points - a 2D array of (x,y) coordinates of data points

# labels - an array of numeric labels in the interval [0..k-1], one for each point

# centers - a 2D array of (x, y) coordinates of cluster centers

# title - title of the plot



def clustering_scatterplot(points, labels, centers, title):

    # plot the examples, i.e. the data points

    

    n_clusters = np.unique(labels).size

    for i in range(n_clusters):

        h = plt.scatter(points[labels==i,0],

                        points[labels==i,1], 

                        c=colors[i%colors.size],

                        label = 'cluster '+str(i))



    # plot the centers of the clusters

    if centers is not None:

        plt.scatter(centers[:,0], centers[:,1], c='r', marker='*', s=500)



    _ = plt.title(title)

    _ = plt.legend()

    _ = plt.xlabel('x')

    _ = plt.ylabel('y')
k = 3

clustered_data_sklearn = cluster.KMeans(n_clusters=k, n_init=10, max_iter=300).fit(data)
# append the cluster centers to the dataset

data_and_centers = np.r_[data,clustered_data_sklearn.cluster_centers_]
# project both th data and the k-Means cluster centers to a 2D space

XYcoordinates = manifold.MDS(n_components=2).fit_transform(data_and_centers)

print("transformation complete")
# plot the transformed examples and the centers

# use the cluster assignment to colour the examples

clustering_scatterplot(points=XYcoordinates[:-k,:], 

                       labels=clustered_data_sklearn.labels_, 

                       centers=XYcoordinates[-k:,:], 

                       title='MDS')
# project both th data and the k-Means cluster centers to a 2D space

XYcoordinates = manifold.TSNE(n_components=2).fit_transform(data_and_centers)

print("transformation complete")
# plot the transformed examples and the centers

# use the cluster assignment to colour the examples

# plot the transformed examples and the centers

# use the cluster assignment to colour the examples

clustering_scatterplot(points=XYcoordinates[:-k,:], 

                       labels=clustered_data_sklearn.labels_,

                       centers=XYcoordinates[-k:,:], 

                       title='TSNE')
df['cluster'] = pd.Series(clustered_data_sklearn.labels_, index=df.index)
df.head()
df.tail()
df.groupby('cluster').mean()