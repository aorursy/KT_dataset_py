%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns; sns.set()  # for plot styling

import numpy as np

import numpy as np

import pandas as pd

from sklearn.preprocessing import scale

from sklearn.preprocessing import LabelEncoder

from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

X, y_true = make_blobs(n_samples=769,

                       cluster_std=0.60, random_state=0)

plt.scatter(X[:, 0], X[:, 1], s=50);

print(X)
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4)

kmeans.fit(X)

y_kmeans = kmeans.predict(X)

print(y_kmeans)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_

plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);