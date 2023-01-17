# importing the libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# getting the dataset

df = pd.read_csv('../input/Mall_Customers.csv')
df.head() # first look at the dataset
df.shape # shape of dataset
df.isna().sum() # finding the null elements in the dataset
sns.countplot(data=df, x="Gender")
_ = plt.hist(data=df, x='Age', bins=[10, 20, 30, 40, 50, 60, 70, 80], color=['green'])

_ = plt.xlabel("Age")
_ = plt.hist(data=df, x='Annual Income (k$)', bins=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140], color=['black'])

_ = plt.xlabel("Annual Income (k$)")
X = df.drop(columns=['CustomerID', 'Gender', 'Annual Income (k$)'])
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=None, algorithm='auto').fit(X)

labels = kmeans.labels_

centroids = kmeans.cluster_centers_
x = df["Age"]

y = df["Spending Score (1-100)"]



plt.scatter(x, y, c=labels)

plt.scatter(centroids[:, 0], centroids[:, 1], color='red')

_ = plt.xlabel('Age')

_ = plt.ylabel('Spending Score (1-100)')
X_2 = df.drop(columns=['CustomerID', 'Gender', 'Age'])
kmeans_2 = KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=None, algorithm='auto').fit(X_2)

labels_2 = kmeans_2.labels_

centroids_2 = kmeans_2.cluster_centers_
x = df['Annual Income (k$)']

y = df['Spending Score (1-100)']



plt.scatter(x, y, c=labels_2)

plt.scatter(centroids_2[:, 0], centroids_2[:, 1], color='red')

_ = plt.xlabel('Annual Income (k$)')

_ = plt.ylabel('Spending Score (1-100)')