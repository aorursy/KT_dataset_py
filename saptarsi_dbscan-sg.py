# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install https://github.com/scikit-learn-contrib/scikit-learn-extra/archive/master.zip
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
colors = ['royalblue','red','deeppink', 'maroon', 'mediumorchid', 'tan', 'forestgreen', 'olive', 'goldenrod', 'lightcyan', 'navy']
vectorizer = np.vectorize(lambda x: colors[x % len(colors)])
X = np.array([[1, 2], [4, 2], [3, 3],[8, 7], [9, 9],[10,10], [25, 80]])
plt.scatter(X[:,0],X[:,1])
def clust_plot(noise):
  plt.figure()
  plt.subplot(131)
  plt.gca().set_title('DBSCAN')
  plt.scatter(X[:,0], X[:,1],c=vectorizer(y+noise))
  plt.subplot(132)
  plt.gca().set_title('k-Means')
  plt.scatter(X[:,0], X[:,1],c=vectorizer(y_kmeans))
  plt.subplot(133)
  plt.gca().set_title('k-Medoid')
  plt.scatter(X[:,0], X[:,1],c=vectorizer(y_kmed))
clustering = DBSCAN(eps=3, min_samples=2).fit(X)
y=clustering.labels_
kmeans = KMeans(n_clusters = 2, init = 'random', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)
kMedoids = KMedoids(n_clusters = 2, random_state = 0)
kMedoids.fit(X)
y_kmed = kMedoids.fit_predict(X)
y
clust_plot(1)
plt.scatter(X[:,0], X[:,1],c=vectorizer(y_kmeans))
#plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'C3')
from sklearn.datasets.samples_generator import make_blobs
X, y = make_blobs(n_samples=300, centers=3, cluster_std=.75, random_state=0)
plt.scatter(X[:,0], X[:,1])
from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=10)
nbrs = neigh.fit(X)
distances, indices = nbrs.kneighbors(X)
distances
distances = np.sort(distances, axis=0)
dist4 = distances[:,3]
dist10 = distances[:,9]
plt.plot(dist4)
plt.plot(dist10)
clustering = DBSCAN(eps=0.4, min_samples=6).fit(X)
y=clustering.labels_
y
plt.scatter(X[:,0], X[:,1], c=vectorizer(y))
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=1500, noise=0.05)
plt.scatter(X[:,0], X[:,1],c=vectorizer(y))
kmeans = KMeans(n_clusters = 2, init = 'random', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)
clustering = DBSCAN(eps=0.1, min_samples=4).fit(X)
y=clustering.labels_
kMedoids.fit(X)
y_kmed = kMedoids.fit_predict(X)
y
clust_plot(0)
from sklearn.datasets import make_circles
X, y = make_circles(n_samples=200, noise=0.01)
plt.scatter(X[:,0], X[:,1],c=vectorizer(y))
kmeans = KMeans(n_clusters = 2, init = 'random', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)
clustering = DBSCAN(eps=0.15, min_samples=4).fit(X)
y=clustering.labels_
kMedoids.fit(X)
y_kmed = kMedoids.fit_predict(X)
y
clust_plot(0)