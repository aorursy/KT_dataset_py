# load up requirements
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# load data
data = pd.read_csv('../input/Iris.csv')
# Pairplot of the data
g = sns.pairplot(data, hue='Species')

# Subset and Normalize

col_names = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
my_data = data[col_names]

# Normalize data
from sklearn import preprocessing

def norm(data):
  min_max_scaler = preprocessing.MinMaxScaler()
  norm_data = min_max_scaler.fit_transform(data)
  df_normalized = pd.DataFrame(norm_data)
  return df_normalized

testing_data = norm(my_data)

# Code Ripped from TensorFlow docs. Hacked for correctness.


def input_fn():
  return tf.train.limit_epochs(
      tf.convert_to_tensor(tf.cast(testing_data, tf.float32)), num_epochs=1)

num_clusters = 3
kmeans = tf.contrib.factorization.KMeansClustering(
    num_clusters=num_clusters, use_mini_batch=False)

# train
num_iterations = 10
previous_centers = None
for _ in range(num_iterations):
  kmeans.train(input_fn)
  cluster_centers = kmeans.cluster_centers()
  if previous_centers is not None:
    print('delta:', cluster_centers - previous_centers)
  previous_centers = cluster_centers
  print('score:', kmeans.score(input_fn))
print('cluster centers:', cluster_centers)

# map the input points to their clusters
cluster_indices = list(kmeans.predict_cluster_index(input_fn))
for i, point in enumerate(testing_data):
  cluster_index = cluster_indices[i]
  center = cluster_centers[cluster_index]
  print('point:', point, 'is in cluster', cluster_index, 'centered at', center)
# Plot Species Clusters

testing_data['cluster_index'] = cluster_indices

# Get our column names back
testing_data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'cluster_index']

k = sns.pairplot(testing_data, hue="cluster_index", hue_order=[1,2,0], vars=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
k.fig.suptitle("K-Means Species Identification", y=1.0)
# Custom Evaluation

# 0 -> Virginica , 1 -> Setosa, 2 -> Versicolor

# Extract cluster_index column
clus = testing_data['cluster_index']

# Replace cluster index with species
clus = clus.replace([0,1,2], ['Iris-virginica', 'Iris-setosa', 'Iris-versicolor'])

# Extract species from original set
species = data['Species']

# Compare
my_bools = species == clus
err = sum(my_bools == False)

print("K-Means clustering incorrectly labeled " + str(err) + " out of " + str(len(clus)) + " flowers as the wrong species.\n")
print("Success rate: " + str(100-err/len(clus) * 100) + "%")

# More visualization
import itertools

final = pd.DataFrame.copy(testing_data)


final['species'] = clus
final['right'] = my_bools

g = sns.pairplot(final, hue="right", palette=["red", "green"],vars=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'] )
g.fig.suptitle("Incorrectly Identified Flowers from the K-Means Model", y=1.0)

  
 