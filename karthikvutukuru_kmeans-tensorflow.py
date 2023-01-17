# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
points_n = 50000

clusters_n = 5

iteration_n = 100

seed = 123

np.random.seed(seed)

tf.random.set_seed(seed)
# Create Random data points and initial centroids

points = np.random.uniform(0, 10, (points_n, 2))

centroids = tf.slice(tf.random.shuffle(points), [0, 0], [clusters_n, -1])
centroids
plt.scatter(points[:, 0], points[:, 1], s=50, alpha=0.5)

plt.plot(centroids[:, 0], centroids[:,1], 'kx', markersize=15)
def closest_centroids(points, centroids):

    distances = tf.reduce_sum(tf.square(tf.subtract(points, centroids[:, None])), 2)

    assignments = tf.argmin(distances)

    return assignments
def move_centroids(points, closest, centroids):

    return np.array([points[closest==k].mean(axis=0) for k in range(centroids.shape[0])])

for _ in range(iteration_n):

    closest = closest_centroids(points, centroids)

    centroids = move_centroids(points, closest, centroids)

    
print("centroids", centroids)
plt.scatter(points[:, 0], points[:, 1], c=closest, s=50, alpha=0.5)

plt.plot(centroids[:, 0], centroids[:, 1], 'kx', markersize=15)

plt.show()