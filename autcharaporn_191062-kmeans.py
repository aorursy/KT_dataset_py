import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

dataset = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')

X = dataset.iloc[:, [3, 4]].values

print(X)
plt.scatter(X[:,0],X[:,1],s = 10, marker  = "o", c = 'blue')

plt.title('Clusters of customers')

plt.xlabel('Annual Income (k$)')

plt.ylabel('Spending Score (1-100)')

plt.legend()

plt.show()
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)

y_kmeans = kmeans.fit_predict(X)
#	Visualising the	clusters

plt.scatter(X[y_kmeans ==	0,	0],	X[y_kmeans ==	0,	1],	s	=	10,	c	=	'red',	label	=	'Cluster	1')

plt.scatter(X[y_kmeans ==	1,	0],	X[y_kmeans ==	1,	1],	s	=	10,	c	=	'blue',	label	=	'Cluster	2')

plt.scatter(X[y_kmeans ==	2,	0],	X[y_kmeans ==	2,	1],	s	=	10,	c	=	'green',	label	=	'Cluster	3')

plt.scatter(X[y_kmeans ==	3,	0],	X[y_kmeans ==	3,	1],	s	=	10,	c	=	'cyan',	label	=	'Cluster	4')

plt.scatter(X[y_kmeans ==	4,	0],	X[y_kmeans ==	4,	1],	s	=	10,	c	=	'magenta',	label	=	'Cluster	5')

plt.scatter(kmeans.cluster_centers_[:,	0],	kmeans.cluster_centers_[:,	1],	s	=	30,	marker	=	"x",	c	=	'black',	label	=	'Centroids')

plt.title('Clusters	of	customers')

plt.xlabel('Annual	Income	(k$)')

plt.ylabel('Spending	Score	(1-100)')

plt.legend()

plt.show()
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)

y_kmeans = kmeans.fit_predict(X)



#	Visualising the	clusters

plt.scatter(X[y_kmeans ==	0,	0],	X[y_kmeans ==	0,	1],	s	=	10,	c	=	'red',	label	=	'Cluster	1')

plt.scatter(X[y_kmeans ==	1,	0],	X[y_kmeans ==	1,	1],	s	=	10,	c	=	'blue',	label	=	'Cluster	2')

plt.scatter(X[y_kmeans ==	2,	0],	X[y_kmeans ==	2,	1],	s	=	10,	c	=	'green',	label	=	'Cluster	3')

plt.scatter(X[y_kmeans ==	3,	0],	X[y_kmeans ==	3,	1],	s	=	10,	c	=	'cyan',	label	=	'Cluster	4')

plt.scatter(X[y_kmeans ==	4,	0],	X[y_kmeans ==	4,	1],	s	=	10,	c	=	'magenta',	label	=	'Cluster	5')

plt.scatter(kmeans.cluster_centers_[:,	0],	kmeans.cluster_centers_[:,	1],	s	=	30,	marker	=	"x",	c	=	'black',	label	=	'Centroids')

plt.title('Clusters	of	customers')

plt.xlabel('Annual	Income	(k$)')

plt.ylabel('Spending	Score	(1-100)')

plt.legend()

plt.show()
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)

y_kmeans = kmeans.fit_predict(X)



#	Visualising the	clusters

plt.scatter(X[y_kmeans ==	0,	0],	X[y_kmeans ==	0,	1],	s	=	10,	c	=	'red',	label	=	'Cluster	1')

plt.scatter(X[y_kmeans ==	1,	0],	X[y_kmeans ==	1,	1],	s	=	10,	c	=	'blue',	label	=	'Cluster	2')

plt.scatter(X[y_kmeans ==	2,	0],	X[y_kmeans ==	2,	1],	s	=	10,	c	=	'green',	label	=	'Cluster	3')

plt.scatter(X[y_kmeans ==	3,	0],	X[y_kmeans ==	3,	1],	s	=	10,	c	=	'cyan',	label	=	'Cluster	4')

plt.scatter(X[y_kmeans ==	4,	0],	X[y_kmeans ==	4,	1],	s	=	10,	c	=	'magenta',	label	=	'Cluster	5')

plt.scatter(kmeans.cluster_centers_[:,	0],	kmeans.cluster_centers_[:,	1],	s	=	30,	marker	=	"x",	c	=	'black',	label	=	'Centroids')

plt.title('Clusters	of	customers')

plt.xlabel('Annual	Income	(k$)')

plt.ylabel('Spending	Score	(1-100)')

plt.legend()

plt.show()
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = 42)

y_kmeans = kmeans.fit_predict(X)



#	Visualising the	clusters

plt.scatter(X[y_kmeans ==	0,	0],	X[y_kmeans ==	0,	1],	s	=	10,	c	=	'red',	label	=	'Cluster	1')

plt.scatter(X[y_kmeans ==	1,	0],	X[y_kmeans ==	1,	1],	s	=	10,	c	=	'blue',	label	=	'Cluster	2')

plt.scatter(X[y_kmeans ==	2,	0],	X[y_kmeans ==	2,	1],	s	=	10,	c	=	'green',	label	=	'Cluster	3')

plt.scatter(X[y_kmeans ==	3,	0],	X[y_kmeans ==	3,	1],	s	=	10,	c	=	'cyan',	label	=	'Cluster	4')

plt.scatter(X[y_kmeans ==	4,	0],	X[y_kmeans ==	4,	1],	s	=	10,	c	=	'magenta',	label	=	'Cluster	5')

plt.scatter(kmeans.cluster_centers_[:,	0],	kmeans.cluster_centers_[:,	1],	s	=	30,	marker	=	"x",	c	=	'black',	label	=	'Centroids')

plt.title('Clusters	of	customers')

plt.xlabel('Annual	Income	(k$)')

plt.ylabel('Spending	Score	(1-100)')

plt.legend()

plt.show()
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 7, init = 'k-means++', random_state = 42)

y_kmeans = kmeans.fit_predict(X) 



#	Visualising the	clusters

plt.scatter(X[y_kmeans ==	0,	0],	X[y_kmeans ==	0,	1],	s	=	10,	c	=	'red',	label	=	'Cluster	1')

plt.scatter(X[y_kmeans ==	1,	0],	X[y_kmeans ==	1,	1],	s	=	10,	c	=	'blue',	label	=	'Cluster	2')

plt.scatter(X[y_kmeans ==	2,	0],	X[y_kmeans ==	2,	1],	s	=	10,	c	=	'green',	label	=	'Cluster	3')

plt.scatter(X[y_kmeans ==	3,	0],	X[y_kmeans ==	3,	1],	s	=	10,	c	=	'cyan',	label	=	'Cluster	4')

plt.scatter(X[y_kmeans ==	4,	0],	X[y_kmeans ==	4,	1],	s	=	10,	c	=	'magenta',	label	=	'Cluster	5')

plt.scatter(kmeans.cluster_centers_[:,	0],	kmeans.cluster_centers_[:,	1],	s	=	30,	marker	=	"x",	c	=	'black',	label	=	'Centroids')

plt.title('Clusters	of	customers')

plt.xlabel('Annual	Income	(k$)')

plt.ylabel('Spending	Score	(1-100)')

plt.legend()

plt.show()