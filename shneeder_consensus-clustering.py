from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn import datasets as ds
from matplotlib import pyplot as plt

iris = ds.load_iris()
iris_data = ds.load_iris().data 

kmeans = KMeans(n_clusters=3, init='k-means++').fit_predict(iris_data)
spectral = SpectralClustering(n_clusters=3).fit_predict(iris_data)

print('K-means:') 
print(kmeans)
print('SpectralClustering:') 
print(spectral)
print('Classification:')
print(ds.load_iris().target)

match = [i for i, j in zip(kmeans, spectral) if i == j]

print(len(match))
print(match)
x_index = 0
y_index = 1
plt.scatter(iris.data[:, x_index], iris.data[:, y_index], c=iris.target)
plt.xlabel(iris.feature_names[x_index])
plt.ylabel(iris.feature_names[y_index])
print('----')
#plt.scatter(iris.data[:, x_index], iris.data[:, y_index], c=kmeans)
fig, axs = plt.subplots(2, 2, figsize=(5, 5))
axs[0, 0].scatter(iris.data[:, x_index], iris.data[:, y_index], c=iris.target)
axs[0, 1].scatter(iris.data[:, x_index], iris.data[:, y_index], c=kmeans)
axs[1, 0].scatter(iris.data[:, x_index], iris.data[:, y_index], c=spectral)
#axs[1, 1].scatter(iris.data[:, x_index], iris.data[:, y_index], c=match)