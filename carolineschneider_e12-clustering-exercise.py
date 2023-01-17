%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from scipy.spatial.distance import cdist

from mpl_toolkits.mplot3d import Axes3D

from sklearn.datasets import make_blobs

from numpy import genfromtxt

plt.rcParams['figure.figsize'] = (16, 9)

plt.style.use('ggplot')





df_orig = pd.read_csv('/kaggle/input/edlich-kmeans-A0.csv')

df_orig.head()
df = pd.read_csv('/kaggle/input/edlich-kmeans-A0.csv', header=None, skiprows=1)

df.head()
col_0 = df[0].values

col_1 = df[1].values

col_2 = df[2].values



fig = plt.figure()

ax = Axes3D(fig)

ax.scatter(col_0, col_1, col_2)
ks = []

distances = []

X = df.values



for k in range(1, 20):

    kmeans = KMeans(n_clusters=k)

    kmeans = kmeans.fit(X)

    labels = kmeans.predict(X)

    centers = kmeans.cluster_centers_

    distance = 0

    for i in range(0, len(labels)):

        c = centers[labels[i]]

        p = X[i]

        distance += np.linalg.norm(p-c)

    ks.append(k)

    distances.append(distance)

plt.plot(ks, distances)
kmeans = KMeans(n_clusters=5)

kmeans = kmeans.fit(X)

labels = kmeans.predict(X)

centers = kmeans.cluster_centers_

for i in range(0, len(labels)):

    print(f"cluster {labels[i]}: data_id: {i}")
fig = plt.figure()

ax = Axes3D(fig)

ax.scatter(col_0, col_1, col_2, c = labels)

ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], marker='*', c='#2f4f4f', s=1000)