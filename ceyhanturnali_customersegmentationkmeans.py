import numpy as np
import pandas as pd
cust=pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
cust.head(10)
cust.info()
cust.isna().sum()
X=cust[['Annual Income (k$)','Spending Score (1-100)']]
X.head(5)

from sklearn import preprocessing
X=preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]
from sklearn.cluster import KMeans
array=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=0)
    kmeans.fit(X)
    array.append(kmeans.inertia_)
import matplotlib.pyplot as plt
plt.plot(range(1,11),array)
plt.xlabel('#ofcluster')
from sklearn.cluster import KMeans
kmeans=KMeans(init='k-means++',n_clusters=5,n_init=12)
kmeansp=kmeans.fit_predict(X)
labels=kmeans.labels_
print(labels)
plt.scatter(X[kmeansp == 0, 0], X[kmeansp == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[kmeansp == 1, 0], X[kmeansp == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[kmeansp == 2, 0], X[kmeansp == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[kmeansp == 3, 0], X[kmeansp == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[kmeansp == 4, 0], X[kmeansp == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()