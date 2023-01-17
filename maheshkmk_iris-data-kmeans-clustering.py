import pandas as pd

import numpy as np

from sklearn.datasets import load_iris



ir = load_iris()

# adding column names for iris data

iris = pd.DataFrame(ir.data, columns= (ir.feature_names))

iris.head()
# removing two columns and going to work on other two columns

iris.drop(['sepal length (cm)','sepal width (cm)'], axis='columns',  inplace=True)

iris.head()
from sklearn.preprocessing import MinMaxScaler

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

%matplotlib inline
#scatter plot with existing data

plt.scatter(iris['petal length (cm)'], iris['petal width (cm)'] )
km = KMeans(n_clusters= 2)

km

y_pre = km.fit_predict(iris[['petal length (cm)','petal width (cm)']])
# adding new column as cluster with predicted cluster data

iris['cluster'] = y_pre

iris.head()
iris1 = iris[iris['cluster']==0] 

iris2 = iris[iris['cluster']==1] 
plt.scatter(iris1['petal length (cm)'], iris1['petal width (cm)'] )

plt.scatter(iris2['petal length (cm)'], iris2['petal width (cm)'] )
scaler = MinMaxScaler()

scaler.fit(iris[['petal width (cm)']])

iris['petal width (cm)'] = scaler.transform(iris[['petal width (cm)']])





scaler = MinMaxScaler()

scaler.fit(iris[['petal length (cm)']])

iris['petal length (cm)'] = scaler.transform(iris[['petal length (cm)']])



iris.head()
km = KMeans(n_clusters= 2)

km

y_pre = km.fit_predict(iris[['petal length (cm)','petal width (cm)']])
iris['cluster'] = y_pre

iris.head()
iris1 = iris[iris['cluster']==0] 

iris2 = iris[iris['cluster']==1] 
plt.scatter(iris1['petal length (cm)'], iris1['petal width (cm)'], label = 'petal width (cm)' )

plt.scatter(iris2['petal length (cm)'], iris2['petal width (cm)'], label = 'petal width (cm)')

plt.legend()
# to check centroid values for clusters 

km.cluster_centers_
plt.figure(figsize = (12,6))

plt.scatter(iris1['petal length (cm)'], iris1['petal width (cm)'], label = 'petal width (cm)' )

plt.scatter(iris2['petal length (cm)'], iris2['petal width (cm)'], label = 'petal width (cm)')

plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1] , color = 'r', marker = '*', label ='centroid', s = 100 )

plt.legend()
k_rng = range(1,10)

ssr = []

for k in k_rng:

    km = KMeans(n_clusters= k)

    km.fit(iris)

    ssr.append(km.inertia_)
# to see ssr values

ssr
plt.figure(figsize = (12,6))

plt.plot(k_rng, ssr)

plt.ylabel('sum of squared error', fontsize = 14)

plt.xlabel('Range', fontsize = 14)