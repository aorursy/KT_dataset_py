#Import relevant libraries 

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

from sklearn.cluster import KMeans
# Load the data

data = pd.read_csv(r'../input/iris-dataset.csv')

# Check the data

data
# Scatter plot based on sepal_length and sepal_width

plt.scatter(data['sepal_length'],data['sepal_width'])

plt.xlabel('Lenght of sepal')

plt.ylabel('Width of sepal')

plt.show()
# Create variable to contain the data for clustering

x = data.copy()

# Create a k-means object with 2 clusters

kmeans = KMeans(2)

# Fit the data

kmeans.fit(x)
# Create a copy of data, to see clusters next to the original data

clusters = data.copy()

# Predict the cluster for each observation

clusters['cluster_pred']=kmeans.fit_predict(x)
# create a scatter plot based on two corresponding features (sepal_length and sepal_width; OR petal_length and petal_width)

plt.scatter(clusters['sepal_length'], clusters['sepal_width'], c= clusters ['cluster_pred'], cmap = 'rainbow')
# Import preprocessing module from sklearn

from sklearn import preprocessing



# Scale the data 

x_scaled = preprocessing.scale(data)

x_scaled
wcss = []

cl_num = 10

for i in range (1,cl_num):

    kmeans= KMeans(i)

    kmeans.fit(x_scaled)

    wcss_iter = kmeans.inertia_

    wcss.append(wcss_iter)

wcss
#Determine best number of clusters

number_clusters = range(1,cl_num)

plt.plot(number_clusters, wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('Within-cluster Sum of Squares')