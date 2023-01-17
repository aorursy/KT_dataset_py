import numpy as np
from scipy import ndimage
from time import time
from sklearn import datasets, manifold 
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl
%matplotlib inline
iris = datasets.load_iris()
X,y = iris.data[:,:2], iris.target


num_clusters = 10
model = KMeans(n_clusters=num_clusters)
model.fit(X)

labels = model.labels_
cluster_centers = model.cluster_centers_

plt.scatter(X[:,0], X[:,1],c=labels.astype(np.float),edgecolors='black')
plt.scatter(cluster_centers[:,0], cluster_centers[:,1], c = np.arange(num_clusters), marker = '^', s = 150,edgecolors='black')
plt.show()

