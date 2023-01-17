import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.cluster import KMeans
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
countries=pd.read_csv(r'/kaggle/input/country-geotags/Countries-exercise.csv')

countries.head()
countries.shape
countries.info()
plt.figure(figsize=(8,8))

sns.scatterplot(x="Longitude",y="Latitude",data=countries)

plt.xlabel('Longitude')

plt.ylabel('Latitude')

plt.title("Data with 0 clusters") 

plt.xlim(-180,180)

plt.ylim(-90, 90)

plt.show()

map = countries.iloc[:,1:]

map.head()
country=countries.copy()

country.head()
kmeans = KMeans(2)
kmeans.fit(map)
clusters=kmeans.fit_predict(map)

clusters
clusters_data=country.copy()

clusters_data['Cluster_2']=clusters

clusters_data.head()
plt.figure(figsize=(8,8))

plt.scatter(countries["Longitude"],countries["Latitude"],c=clusters_data['Cluster_2'],cmap='rainbow')

plt.xlabel('Longitude')

plt.ylabel('Latitude')

plt.title("Data with 2 clusters") 

plt.xlim(-180,180)

plt.ylim(-90, 90)

plt.show()

kmeans = KMeans(3)

kmeans.fit(map)
clusters=kmeans.fit_predict(map)

clusters_data['Cluster_3']=clusters

plt.figure(figsize=(8,8))

plt.scatter(countries["Longitude"],countries["Latitude"],c=clusters_data['Cluster_3'],cmap='rainbow')

plt.xlabel('Longitude')

plt.ylabel('Latitude')

plt.title("Data with 3 clusters") 

plt.xlim(-180,180)

plt.ylim(-90, 90)

plt.show()
kmeans = KMeans(4)

kmeans.fit(map)
clusters=kmeans.fit_predict(map)

clusters_data['Cluster_4']=clusters

plt.figure(figsize=(8,8))

plt.scatter(countries["Longitude"],countries["Latitude"],c=clusters_data['Cluster_4'],cmap='rainbow')

plt.xlabel('Longitude')

plt.ylabel('Latitude')

plt.title("Data with 4 clusters") 

plt.xlim(-180,180)

plt.ylim(-90, 90)

plt.show()
kmeans = KMeans(5)

kmeans.fit(map)
clusters=kmeans.fit_predict(map)

clusters_data['Cluster_5']=clusters

plt.figure(figsize=(8,8))

plt.scatter(countries["Longitude"],countries["Latitude"],c=clusters_data['Cluster_5'],cmap='rainbow')

plt.xlabel('Longitude')

plt.ylabel('Latitude')

plt.title("Data with 5 clusters") 

plt.xlim(-180,180)

plt.ylim(-90, 90)

plt.show()
kmeans = KMeans(6)

kmeans.fit(map)
clusters=kmeans.fit_predict(map)

clusters_data['Cluster_6']=clusters

plt.figure(figsize=(8,8))

plt.scatter(countries["Longitude"],countries["Latitude"],c=clusters_data['Cluster_6'],cmap='rainbow')

plt.xlabel('Longitude')

plt.ylabel('Latitude')

plt.title("Data with 6 clusters") 

plt.xlim(-180,180)

plt.ylim(-90, 90)

plt.show()
kmeans = KMeans(7)

kmeans.fit(map)
clusters=kmeans.fit_predict(map)

clusters_data['Cluster_7']=clusters

plt.figure(figsize=(8,8))

plt.scatter(countries["Longitude"],countries["Latitude"],c=clusters_data['Cluster_7'],cmap='rainbow')

plt.xlabel('Longitude')

plt.ylabel('Latitude')

plt.title("Data with 7 clusters") 

plt.xlim(-180,180)

plt.ylim(-90, 90)

plt.show()
kmeans = KMeans(8)

kmeans.fit(map)
clusters=kmeans.fit_predict(map)

clusters_data['Cluster_8']=clusters

plt.figure(figsize=(8,8))

plt.scatter(countries["Longitude"],countries["Latitude"],c=clusters_data['Cluster_8'],cmap='rainbow')

plt.xlabel('Longitude')

plt.ylabel('Latitude')

plt.title("Data with 8 clusters") 

plt.xlim(-180,180)

plt.ylim(-90, 90)

plt.show()
kmeans = KMeans(9)

kmeans.fit(map)
clusters=kmeans.fit_predict(map)

clusters_data['Cluster_9']=clusters

plt.figure(figsize=(8,8))

plt.scatter(countries["Longitude"],countries["Latitude"],c=clusters_data['Cluster_9'],cmap='rainbow')

plt.xlabel('Longitude')

plt.ylabel('Latitude')

plt.title("Data with 9 clusters") 

plt.xlim(-180,180)

plt.ylim(-90, 90)

plt.show()
plt.figure(figsize=(25, 25))

plt.subplot(4,2,1)

plt.scatter(countries["Longitude"],countries["Latitude"],c=clusters_data['Cluster_2'],cmap='rainbow')

plt.xlabel('Longitude')

plt.ylabel('Latitude')

plt.title("Data with 2 clusters") 

plt.xlim(-180,180)

plt.ylim(-90, 90)

plt.subplot(4,2,2)

plt.scatter(countries["Longitude"],countries["Latitude"],c=clusters_data['Cluster_3'],cmap='rainbow')

plt.xlabel('Longitude')

plt.ylabel('Latitude')

plt.title("Data with 3 clusters") 

plt.xlim(-180,180)

plt.ylim(-90, 90)

plt.subplot(4,2,3)

plt.scatter(countries["Longitude"],countries["Latitude"],c=clusters_data['Cluster_4'],cmap='rainbow')

plt.xlabel('Longitude')

plt.ylabel('Latitude')

plt.title("Data with 4 clusters") 

plt.xlim(-180,180)

plt.ylim(-90, 90)

plt.subplot(4,2,4)

plt.scatter(countries["Longitude"],countries["Latitude"],c=clusters_data['Cluster_5'],cmap='rainbow')

plt.xlabel('Longitude')

plt.ylabel('Latitude')

plt.title("Data with 5 clusters") 

plt.xlim(-180,180)

plt.ylim(-90, 90)

plt.subplot(4,2,5)

plt.scatter(countries["Longitude"],countries["Latitude"],c=clusters_data['Cluster_6'],cmap='rainbow')

plt.xlabel('Longitude')

plt.ylabel('Latitude')

plt.title("Data with 6 clusters") 

plt.xlim(-180,180)

plt.ylim(-90, 90)

plt.subplot(4,2,6)

plt.scatter(countries["Longitude"],countries["Latitude"],c=clusters_data['Cluster_7'],cmap='rainbow')

plt.xlabel('Longitude')

plt.ylabel('Latitude')

plt.title("Data with 7 clusters") 

plt.xlim(-180,180)

plt.ylim(-90, 90)

plt.subplot(4,2,7)

plt.scatter(countries["Longitude"],countries["Latitude"],c=clusters_data['Cluster_8'],cmap='rainbow')

plt.xlabel('Longitude')

plt.ylabel('Latitude')

plt.title("Data with 8 clusters") 

plt.xlim(-180,180)

plt.ylim(-90, 90)

plt.subplot(4,2,8)

plt.scatter(countries["Longitude"],countries["Latitude"],c=clusters_data['Cluster_9'],cmap='rainbow')

plt.xlabel('Longitude')

plt.ylabel('Latitude')

plt.title("Data with 9 clusters") 

plt.xlim(-180,180)

plt.ylim(-90, 90)

plt.show()
