import numpy as np

import pandas as pd

import seaborn as sns

sns.set_palette('husl') #สีสวยดูง่าย แนะนำครับ

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

%matplotlib inline

data = pd.read_csv('../input/Iris.csv', index_col='Id')
data.head()
data.info()
data['Species'].value_counts()
g = sns.pairplot(data, hue='Species', markers=['+', 'o', 'x'])

plt.show()
g = sns.boxplot(y='Species', x='SepalLengthCm', data=data, orient = 'h')

plt.show()

g = sns.boxplot(y='Species', x='SepalWidthCm', data=data, orient = 'h')

plt.show()

g = sns.boxplot(y='Species', x='PetalLengthCm', data=data, orient = 'h')

plt.show()

g = sns.boxplot(y='Species', x='PetalWidthCm', data=data, orient = 'h')

plt.show()
from sklearn.cluster import KMeans

X_iris = data.iloc[:,[0,1,2,3]].values
from sklearn.cluster import KMeans

wcss = []



for i in range(1, 11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

    kmeans.fit(X_iris)

    wcss.append(kmeans.inertia_)

    

plt.plot(range(1, 11), wcss)

plt.title('The elbow method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
#Applying kmeans to the dataset / Creating the kmeans classifier

kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

y_kmeans = kmeans.fit_predict(X_iris)
plt.scatter(X_iris[y_kmeans == 0, 0], X_iris[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Iris-setosa')

plt.scatter(X_iris[y_kmeans == 1, 0], X_iris[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Iris-versicolour')

plt.scatter(X_iris[y_kmeans == 2, 0], X_iris[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Iris-virginica')



plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroids')



plt.legend()