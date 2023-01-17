import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os



from sklearn.cluster import KMeans

from yellowbrick.cluster import KElbowVisualizer

from sklearn import metrics



dataset = pd.read_csv('../input/seed-from-uci/Seed_Data.csv')

dataset.head()
dataset.describe(include = "all")
features = dataset.iloc[:, 0:7]

target = dataset.iloc[:, -1]



model = KMeans()

visualizer = KElbowVisualizer(model, k=(1,10))



visualizer.fit(features)    # Fit the data to the visualizer

visualizer.poof()    # Draw/show/poof the data
kmeans = KMeans(n_clusters=3)

kmeans.fit(features)

cluster_labels = kmeans.fit_predict(features)



kmeans.cluster_centers_
silhouette_avg = metrics.silhouette_score(features, cluster_labels)

print ('silhouette coefficient for the above clutering = ', silhouette_avg)
def purity_score(y_true, y_pred):

    # compute contingency matrix (also called confusion matrix)

    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)

    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 



purity = purity_score(target, cluster_labels)

print ('Purity for the above clutering = ', purity)
!pip install pyclustering



from pyclustering.cluster.kmedoids import kmedoids
# Randomly pick 3 indexs from the original sample as the mediods

initial_medoids = [1, 50, 170]



# Create instance of K-Medoids algorithm with prepared centers.

kmedoids_instance = kmedoids(features.values.tolist(), initial_medoids)



# Run cluster analysis.

kmedoids_instance.process()



# predict function is not availble in the release branch yet.

# cluster_labels = kmedoids_instance.predict(features.values)



clusters = kmedoids_instance.get_clusters()



# Prepare cluster labels

cluster_labels = np.zeros([210], dtype=int)

for x in np.nditer(np.asarray(clusters[1])):

   cluster_labels[x] = 1

for x in np.nditer(np.asarray(clusters[2])):

   cluster_labels[x] = 2



cluster_labels
# Mediods found in above clustering, indexes are shouwn below.

kmedoids_instance.get_medoids()
silhouette_avg = metrics.silhouette_score(features, cluster_labels)

print ('silhouette coefficient for the above clutering = ', silhouette_avg)
purity = purity_score(target, cluster_labels)

print ('Purity for the above clutering = ', purity)