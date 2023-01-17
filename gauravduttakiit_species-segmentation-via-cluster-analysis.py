import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

from sklearn.cluster import KMeans
# Load the data

data = pd.read_csv('/kaggle/input/iris-dataset/iris-dataset.csv')

# Check the data

data.head()
# Create a scatter plot based on two corresponding features (sepal_length and sepal_width; OR petal_length and petal_width)

plt.figure(figsize=(10, 10))

plt.scatter(data['sepal_length'],data['sepal_width'])

# Name your axes

plt.xlabel('Lenght of sepal')

plt.ylabel('Width of sepal')

plt.show()
# create a variable which will contain the data for the clustering

x = data.copy()

# create a k-means object with 2 clusters

kmeans = KMeans(2)

# fit the data

kmeans.fit(x)
# create a copy of data, so we can see the clusters next to the original data

clusters = data.copy()

# predict the cluster for each observation

clusters['cluster_pred']=kmeans.fit_predict(x)
# create a scatter plot based on two corresponding features (sepal_length and sepal_width; OR petal_length and petal_width)

plt.figure(figsize=(10, 10))

plt.scatter(clusters['sepal_length'], clusters['sepal_width'], c= clusters ['cluster_pred'], cmap = 'rainbow')

plt.xlabel('sepal_length')

plt.ylabel('sepal_width')

plt.title("sepal_length v/s sepal_width")

plt.show()
# import some preprocessing module

from sklearn import preprocessing



# scale the data for better results

x_scaled = preprocessing.scale(data)

x_scaled
# create a k-means object with 2 clusters

kmeans_scaled = KMeans(2)

# fit the data

kmeans_scaled.fit(x_scaled)
# create a copy of data, so we can see the clusters next to the original data

clusters_scaled = data.copy()

# predict the cluster for each observation

clusters_scaled['cluster_pred']=kmeans_scaled.fit_predict(x_scaled)
# create a scatter plot based on two corresponding features (sepal_length and sepal_width; OR petal_length and petal_width)

plt.figure(figsize=(10, 10))

plt.scatter(clusters_scaled['sepal_length'], clusters_scaled['sepal_width'], c= clusters_scaled ['cluster_pred'], cmap = 'rainbow')

plt.xlabel('sepal_length')

plt.ylabel('sepal_width')

plt.title("sepal_length v/s sepal_width")

plt.show()
wcss = []

# 'cl_num' is a that keeps track the highest number of clusters we want to use the WCSS method for. We have it set at 10 right now, but it is completely arbitrary.

cl_num = 10

for i in range (1,cl_num):

    kmeans= KMeans(i)

    kmeans.fit(x_scaled)

    wcss_iter = kmeans.inertia_

    wcss.append(wcss_iter)

wcss
number_clusters = range(1,cl_num)

plt.plot(number_clusters, wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('Within-cluster Sum of Squares')

plt.show()
kmeans_2 = KMeans(2)

kmeans_2.fit(x_scaled)
# Remember that we are plotting the non-standardized values of the sepal length and width. 

clusters_2 = x.copy()

clusters_2['cluster_pred']=kmeans_2.fit_predict(x_scaled)
plt.figure(figsize=(10, 10))

plt.scatter(clusters_2['sepal_length'], clusters_2['sepal_width'], c= clusters_2 ['cluster_pred'], cmap = 'rainbow')

plt.xlabel('sepal_length')

plt.ylabel('sepal_width')

plt.title("sepal_length v/s sepal_width")

plt.show()
kmeans_3 = KMeans(3)

kmeans_3.fit(x_scaled)
clusters_3 = x.copy()

clusters_3['cluster_pred']=kmeans_3.fit_predict(x_scaled)
plt.figure(figsize=(10, 10))

plt.scatter(clusters_3['sepal_length'], clusters_3['sepal_width'], c= clusters_3 ['cluster_pred'], cmap = 'rainbow')

plt.xlabel('sepal_length')

plt.ylabel('sepal_width')

plt.title("sepal_length v/s sepal_width")

plt.show()
kmeans_5 = KMeans(5)

kmeans_5.fit(x_scaled)
clusters_5 = x.copy()

clusters_5['cluster_pred']=kmeans_5.fit_predict(x_scaled)
plt.figure(figsize=(10, 10))

plt.scatter(clusters_5['sepal_length'], clusters_5['sepal_width'], c= clusters_5 ['cluster_pred'], cmap = 'rainbow')

plt.xlabel('sepal_length')

plt.ylabel('sepal_width')

plt.title("sepal_length v/s sepal_width")

plt.show()
real_data = pd.read_csv(r'/kaggle/input/iris-dataset/iris-with-answers.csv')
real_data['species'].unique()
# We use the map function to change any 'yes' values to 1 and 'no'values to 0. 

real_data['species'] = real_data['species'].map({'setosa':0, 'versicolor':1 , 'virginica':2})
real_data.head()
plt.figure(figsize=(10, 10))

plt.scatter(real_data['sepal_length'], real_data['sepal_width'], c= real_data ['species'], cmap = 'rainbow')

plt.xlabel('sepal_length')

plt.ylabel('sepal_width')

plt.title("sepal_length v/s sepal_width")

plt.show()
plt.figure(figsize=(10, 10))

plt.scatter(real_data['petal_length'], real_data['petal_width'], c= real_data ['species'], cmap = 'rainbow')

plt.xlabel('sepal_length')

plt.ylabel('sepal_width')

plt.title("sepal_length v/s sepal_width")

plt.show()
plt.figure(figsize=(10, 10))

plt.scatter(clusters_3['sepal_length'], clusters_3['sepal_width'], c= clusters_3 ['cluster_pred'], cmap = 'rainbow')

plt.xlabel('sepal_length')

plt.ylabel('sepal_width')

plt.title("sepal_length v/s sepal_width")

plt.show()
plt.figure(figsize=(10, 10))

plt.scatter(clusters_3['petal_length'], clusters_3['petal_width'], c= clusters_3 ['cluster_pred'], cmap = 'rainbow')

plt.xlabel('sepal_length')

plt.ylabel('sepal_width')

plt.title("sepal_length v/s sepal_width")

plt.show()