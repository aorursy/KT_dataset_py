import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn import datasets
# load iris dataset from sklearn library
iris = datasets.load_iris()
# DESCR contains details about iris dataset
print(iris.DESCR)
print('Features:'+str(iris.feature_names))
print('Labels'+str(iris.target_names))
X = iris.data
y = iris.target
print(X.shape)
print(y.shape)
# import library for 3D plotting
from mpl_toolkits import mplot3d

# magic function for interactive plot
%matplotlib notebook
plt.figure('Iris dataset', figsize=(7,5))
ax = plt.axes(projection = '3d')
ax.scatter(X[:,3],X[:,0],X[:,2],c=y);
# import K-means algorithm from sklearn

from sklearn.cluster import KMeans
# set up hyperparameter (number of clusters)

k_means = KMeans(n_clusters=3)
# compute k-means clustering

k_means.fit(X)
# compute cluster centers and predict cluster index for each sample

k_means_predicted = k_means.predict(X)
# calculate the accuracy

accuracy = round((np.mean(k_means_predicted==y))*100)
print('Accuracy:'+str(accuracy))
centroids = k_means.cluster_centers_
target_names = iris.target_names
colors = ['navy', 'turquoise', 'darkorange']
plt.figure('K-Means on Iris Dataset', figsize=(7,7))
ax = plt.axes(projection = '3d')
ax.scatter(X[:,3],X[:,0],X[:,2], c=y , cmap='Set2', s=50)

# color missclassified data

ax.scatter(X[k_means_predicted!=y,3],X[k_means_predicted!=y,0],X[k_means_predicted!=y,2] ,c='b', s=50)

# plot centroids

ax.scatter(centroids[0,3],centroids[0,0],centroids[0,2] ,c='r', s=50, label='centroid')
ax.scatter(centroids[1,3],centroids[1,0],centroids[1,2] ,c='r', s=50)
ax.scatter(centroids[2,3],centroids[2,0],centroids[2,2] ,c='r', s=50)

ax.legend()
