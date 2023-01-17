import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.cluster import KMeans
countries=pd.read_csv(r'/kaggle/input/categorical-country-geotags/Categorical.csv')

countries.head()
plt.figure(figsize=(8,8))

sns.scatterplot(x="Longitude",y="Latitude",data=countries)

plt.xlim(-180,180)

plt.ylim(-90, 90)

plt.xlabel('Longitude')

plt.ylabel('Latitude')

plt.show()
x = countries.iloc[:,1:3]

x.head()
kmeans = KMeans(4)

kmeans.fit(x)
clusters=kmeans.fit_predict(x)



clusters
clusters_data=countries.copy()

clusters_data['Cluster_4']=clusters

clusters_data.head()
clusters=kmeans.fit_predict(x)

clusters_data['Cluster_4']=clusters

plt.figure(figsize=(8,8))

plt.scatter(countries["Longitude"],countries["Latitude"],c=clusters_data['Cluster_4'],cmap='rainbow')

plt.title("Data with 9 clusters") 

plt.xlabel('Longitude')

plt.ylabel('Latitude')

plt.xlim(-180,180)

plt.ylim(-90, 90)

plt.show()

kmeans.inertia_
wcss = []

# 'cl_num' is a that keeps track the highest number of clusters we want to use the WCSS method for.

# Note that 'range' doesn't include the upper boundery

cl_num = 10

for i in range (1,cl_num):

    kmeans= KMeans(i)

    kmeans.fit(x)

    wcss_iter = kmeans.inertia_

    wcss.append(wcss_iter)
wcss
number_clusters = range(1,cl_num)

plt.plot(number_clusters, wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('Within-cluster Sum of Squares')
kmeans = KMeans(2)

kmeans.fit(x)


clusters=kmeans.fit_predict(x)

clusters_data['Cluster_2']=clusters

plt.scatter(countries["Longitude"],countries["Latitude"],c=clusters_data['Cluster_2'],cmap='rainbow')

plt.xlabel('Longitude')

plt.ylabel('Latitude')

plt.title("Data with 2 clusters") 

plt.xlim(-180,180)

plt.ylim(-90, 90)

plt.show()
kmeans = KMeans(3)

kmeans.fit(x)


clusters=kmeans.fit_predict(x)

clusters_data['Cluster_3']=clusters

plt.scatter(countries["Longitude"],countries["Latitude"],c=clusters_data['Cluster_3'],cmap='rainbow')

plt.xlabel('Longitude')

plt.ylabel('Latitude')

plt.title("Data with 3 clusters") 

plt.xlim(-180,180)

plt.ylim(-90, 90)

plt.show()
clusters_data.head()
plt.figure(figsize=(25, 20))

plt.subplot(2,1,1)

plt.scatter(countries["Longitude"],countries["Latitude"],c=clusters_data['Cluster_2'],cmap='rainbow')

plt.xlabel('Longitude')

plt.ylabel('Latitude')

plt.title("Data with 2 clusters") 

plt.xlim(-180,180)

plt.ylim(-90, 90)

plt.subplot(2,1,2)

plt.scatter(countries["Longitude"],countries["Latitude"],c=clusters_data['Cluster_3'],cmap='rainbow')

plt.xlabel('Longitude')

plt.ylabel('Latitude')

plt.title("Data with 3 clusters") 

plt.xlim(-180,180)

plt.ylim(-90, 90)



plt.show()