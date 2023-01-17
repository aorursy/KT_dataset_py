from sklearn import datasets

myiris = datasets.load_iris()

x = myiris.data

y = myiris.target
#Scaling using standard scalar

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(x)

x_scaled = scaler.transform(x)
"""

#Scaling using Min-max scalar

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

x_scaled = scaler.fit_transform(x)

"""
import matplotlib.pyplot as plt

plt.scatter(x_scaled[y == 0, 0], x_scaled[y == 0, 1], s = 100, c = 'red', label = 'Iris-setosa')

plt.scatter(x_scaled[y == 1, 0], x_scaled[y == 1, 1], s = 100, c = 'blue', label = 'Iris-versicolour')

plt.scatter(x_scaled[y == 2, 0], x_scaled[y == 2, 1], s = 100, c = 'green', label = 'Iris-virginica')
from sklearn.cluster import KMeans

wcss = []



for i in range(1, 11):

    kmeans = KMeans(n_clusters = i, init = 'random', max_iter = 300, n_init = 10, random_state = 0)

    kmeans.fit(x_scaled)

    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)

plt.title('The elbow method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')      #within cluster sum of squares

plt.show()
kmeans = KMeans(n_clusters = 3, init = 'random', max_iter = 300, n_init = 10, random_state = 0)

y_kmeans = kmeans.fit_predict(x_scaled)
plt.scatter(x_scaled[y_kmeans == 0, 0], x_scaled[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Iris-setosa')

plt.scatter(x_scaled[y_kmeans == 1, 0], x_scaled[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Iris-versicolour')

plt.scatter(x_scaled[y_kmeans == 2, 0], x_scaled[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Iris-virginica')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroids')

plt.legend()