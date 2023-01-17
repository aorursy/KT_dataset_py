import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

from sklearn import datasets
iris = datasets.load_iris()
iris
type(iris)
iris.keys()
iris['data']
iris['target']
iris['frame']
iris['target_names']
iris['DESCR']
iris['feature_names']
iris['filename']
data = pd.DataFrame(iris.data, columns = iris.feature_names)
data.head()
data.tail()
data.shape
data.size
data.describe()
#Finding null values in data  

data.isnull().sum()
sns.pairplot(data, kind="reg")
# Finding the optimum number of clusters for k-means classification



x = data.iloc[:, [0, 1, 2, 3]].values
x
from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

    kmeans.fit(x)

    wcss.append(kmeans.inertia_)
wcss
# Plotting the results onto a line graph, 

# `allowing us to observe 'The elbow'

plt.plot(range(1, 11), wcss)

plt.title('The elbow method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS') # Within cluster sum of squares

plt.show()
# Applying kmeans to the dataset / Creating the kmeans classifier

kmeans = KMeans(n_clusters = 3, init = 'k-means++',max_iter = 300, n_init = 10, random_state = 0)

y_kmeans = kmeans.fit_predict(x)
#Visualising the clusters

plt.scatter(x[:,0], x[:,1])

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')

plt.show()
# Visualising the clusters - On the first two columns

plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Iris-setosa')

plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Iris-versicolour')

plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1],s = 100, c = 'green', label = 'Iris-virginica')



# Plotting the centroids of the clusters

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroids')



plt.legend()