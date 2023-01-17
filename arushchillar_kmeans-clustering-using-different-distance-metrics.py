import pandas as pd

iris = pd.read_csv('../input/iris/Iris.csv')

iris.head()
X = iris.iloc[:, 1:5].values # feature matrix

y = iris.iloc[:, -1].values # class matrix
from yellowbrick.cluster import KElbowVisualizer

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

%matplotlib inline



# Instantiate the clustering model and visualizer

model = KMeans()

visualizer = KElbowVisualizer(model, k=(1, 11))



visualizer.fit(X)        # Fit the data to the visualizer

visualizer.show()        # Finalize and render the figure

plt.show()
# instatiate KMeans class and set the number of clusters

kmeans = KMeans(n_clusters=3, random_state=10)



# call fit method with data 

km = kmeans.fit_predict(X)



# coordinates of cluster center

centroids = kmeans.cluster_centers_ 



# cluster label for each data point

labels = kmeans.labels_ 
plt.scatter(

    X[km == 0, 0], X[km == 0, 1],

    s=25, c='lightgreen',

    marker='s', edgecolor='black',

    label='cluster 1'

)



plt.scatter(

    X[km == 1, 0], X[km == 1, 1],

    s=25, c='yellow',

    marker='o', edgecolor='black',

    label='cluster 2'

)



plt.scatter(

    X[km == 2, 0], X[km == 2, 1],

    s=25, c='lightblue',

    marker='v', edgecolor='black',

    label='cluster 3'

)



# visualise centroids

plt.scatter(

    centroids[:, 0], centroids[:, 1],

    s=300, marker='*',

    c='red', edgecolor='black',

    label='centroids'

)

plt.legend(scatterpoints=1)

plt.grid()

plt.show()
import numpy as np

from sklearn import metrics



def purity_score(y_true, y_pred):

    # compute contingency matrix (also called confusion matrix)

    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)

    # return purity

    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)



# Report Purity Score

purity = purity_score(y, labels)

print(f"The purity score is {round(purity*100, 2)}%")
!pip install pyclustering
from pyclustering.cluster.kmeans import kmeans

from pyclustering.utils.metric import distance_metric

from pyclustering.cluster.center_initializer import random_center_initializer

from pyclustering.cluster.encoder import type_encoding

from pyclustering.cluster.encoder import cluster_encoder



# define dictionary for distance measures

distance_measures = {'euclidean': 0, 'squared euclidean': 1, 'manhattan': 2, 'chebyshev': 3, 

                    'canberra': 5, 'chi-square': 6}



# function defined to compute purity score using pyclustering for various distance measures

def pyPurity(dist_measure):

    initial_centers = random_center_initializer(X, 3, random_state=5).initialize()

    # instance created for respective distance metric

    instanceKm = kmeans(X, initial_centers=initial_centers, metric=distance_metric(dist_measure))

    # perform cluster analysis

    instanceKm.process()

    # cluster analysis results - clusters and centers

    pyClusters = instanceKm.get_clusters()

    pyCenters = instanceKm.get_centers()

    # enumerate encoding type to index labeling to get labels

    pyEncoding = instanceKm.get_cluster_encoding()

    pyEncoder = cluster_encoder(pyEncoding, pyClusters, X)

    pyLabels = pyEncoder.set_encoding(0).get_clusters()

    return purity_score(y, pyLabels)



# print results

for measure, value in distance_measures.items():

    print(f"The purity score for {measure} distance is {round(pyPurity(value)*100, 2)}%")