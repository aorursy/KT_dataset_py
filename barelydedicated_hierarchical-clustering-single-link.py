

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.cluster.hierarchy import dendrogram, linkage

from scipy.spatial.distance import cdist

from matplotlib import pyplot as plt

from scipy.spatial import distance

import math

%matplotlib inline

np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
c1 = pd.read_csv("../input/C1.csv", names=['x0', 'x1'])
plt.scatter(c1['x0'],c1['x1'])
def single_distance(clusters ,cluster_num):

    print('first cluster | ','second cluster | ', 'distance')

    while len(clusters) is not cluster_num:

        # Clustering           (

        closest_distance=clust_1=clust_2 = math.inf

        # for every cluster (until second last element)

        for cluster_id, cluster in enumerate(clusters[:len(clusters)]): 

            # for each point in each cluster

            for point_id,point in enumerate(cluster): 

                # we only need to compare with clusters after the current one

                for cluster2_id, cluster2 in enumerate(clusters[(cluster_id+1):]): 

                    # go through every point in this prospective cluster as well

                    for point2_id, point2 in enumerate(cluster2):

# if this distance is better than our previous best distance then we are going to overwrite it

                        if distance.euclidean(point,point2) < closest_distance: 

                            # Only used for comparing 

                            closest_distance = distance.euclidean(point,point2)

                # this will be used at the end to figure out which cluster to merge with which

                            clust_1 = cluster_id

                # this cluster will be destroyed by the end

                            clust_2 = cluster2_id+cluster_id+1

               # extend just appends the contents to the list without flattening it out

        print(clust_1,' | ',clust_2, ' | ',closest_distance)

        clusters[clust_1].extend(clusters[clust_2]) 

        # don't need this index anymore, and we have just clustered once more

        clusters.pop(clust_2) 

    return(clusters)
### Hierarchical clustering

def hierarchical(data, cluster_num, metric = 'single'):

    # initialization of clusters at first (every point is a cluster)

    init_clusters=[]

    for index, row in data.iterrows():

        init_clusters.append([[row['x0'], row['x1']]])

    if metric is 'single':

        return single_distance(init_clusters, cluster_num)
clusters = hierarchical(c1,4)

colors = ['blue', 'red', 'purple', 'teal']

for cluster_index, cluster in enumerate(clusters):

    for point_index, point in enumerate(cluster):

        plt.plot([point[0]], [point[1]], marker='o', markersize=3, color=colors[cluster_index])
X = c1.as_matrix()

# generate the linkage matrix

single_link = linkage(X, 'single') # using single link metric to evaluate 'distance' between clusters
from scipy.cluster.hierarchy import cophenet

from scipy.spatial.distance import pdist
c, coph_dists = cophenet(single_link, pdist(X))

c
single_link[0]
single_link[1]
single_link[:20]
# calculate full dendrogram

plt.figure(figsize=(25, 10))

plt.title('Hierarchical Clustering Dendrogram')

plt.xlabel('sample index')

plt.ylabel('distance')

dendrogram(

    single_link,

    leaf_rotation=90.,  # rotates the x axis labels

    leaf_font_size=8.,  # font size for the x axis labels

    color_threshold= .6

)

plt.show()
clusters = hierarchical(c1,4)

colors = ['blue', 'red', 'purple', 'teal']

for cluster_index, cluster in enumerate(clusters):

    for point_index, point in enumerate(cluster):

        plt.plot([point[0]], [point[1]], marker='o', markersize=3, color=colors[cluster_index])