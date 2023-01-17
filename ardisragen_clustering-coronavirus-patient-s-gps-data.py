import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

import seaborn as sns; sns.set()

import csv



path = '/kaggle/input/coronavirusdataset/'

route_data_path = path + 'route.csv'



df_route = pd.read_csv(route_data_path)
df_route.head(10)
df_route.tail(10)
df_route.dropna(axis=0,how='any',subset=['latitude','longitude'],inplace=True)
# Variable with the Longitude and Latitude

X=df_route.loc[:,['id','latitude','longitude']]

X.head(10)
X.tail(10)
K_clusters = range(1,10)

kmeans = [KMeans(n_clusters=i) for i in K_clusters]

Y_axis = df_route[['latitude']]

X_axis = df_route[['longitude']]

score = [kmeans[i].fit(Y_axis).score(Y_axis) for i in range(len(kmeans))]

# Visualize

plt.plot(K_clusters, score)

plt.xlabel('Number of Clusters')

plt.ylabel('Score')

plt.title('Elbow Curve')

plt.show()
kmeans = KMeans(n_clusters = 4, init ='k-means++')

kmeans.fit(X[X.columns[1:3]]) # Compute k-means clustering.

X['cluster_label'] = kmeans.fit_predict(X[X.columns[1:3]])

centers = kmeans.cluster_centers_ # Coordinates of cluster centers.

labels = kmeans.predict(X[X.columns[1:3]]) # Labels of each point

X.plot.scatter(x = 'latitude', y = 'longitude', c=labels, s=50, cmap='viridis')

plt.scatter(centers[:, 0], centers[:, 1], c='black', s=100, alpha=0.5)
X[0:60]