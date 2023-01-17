# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataset = pd.read_csv("/kaggle/input/realstate-cleancsv/Realestate.csv")

dataset.info()
location = dataset.iloc[:, [4, 5]].values

location
import matplotlib.pyplot as plt

plt.scatter(location[:,0], location[:,1], marker = "x", color = 'R', s = 60)

plt.xlabel('X5 latitude')

plt.ylabel('X6 longitude')

plt.show()
# find the optimal number of clusters

from sklearn.cluster import KMeans

wcss = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)

    kmeans.fit(location)

    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.savefig("wcss_NumberofClusters.png")

plt.show()
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 0)

y_kmeans = kmeans.fit_predict(location)

y_kmeans
from matplotlib.colors import ListedColormap

raw_colors = ("red", "green", "blue", "orange", "purple")

colors = ListedColormap(raw_colors)

for i in range(5):

    plt.scatter(location[y_kmeans == i,0], location[y_kmeans == i,1],s= 60,c = colors(i), marker = "x")

X_clusters = kmeans.cluster_centers_[:,0]

Y_clusters = kmeans.cluster_centers_[:,1]

plt.scatter(X_clusters, Y_clusters,s= 60,c= "yellow")

plt.title('Clusters of area')

plt.xlabel('X5 latitude')

plt.ylabel('X6 longitude')

plt.show()
Area=y_kmeans

dataset_new = dataset.copy()

dataset_new['X5 Area']=Area

dataset_new
location_map=dataset_new[['X5 latitude','X6 longitude','X5 Area']]
location_map['color']=location_map['X5 Area'].apply(lambda area:"red" if area==0 else

                                         "green" if area==1 else

                                         "Orange" if area==2 else

                                         "blue" if area==3 else "brown" )



location_map['size']=location_map['X5 Area'].apply(lambda area:6)

import folium

m = folium.Map(location = [24.968, 121.53], zoom_start = 13)

#location=location[0:2000]

for lat,lon,price,color,size in zip(location_map['X5 latitude'],location_map['X6 longitude'],location_map['X5 Area'],location_map['color'],location_map['size']):

     folium.CircleMarker([lat, lon],

                            popup=price,

                            radius=size,

                            color='b',

                            fill=True,

                            fill_opacity=0.7,

                            fill_color=color,

                           ).add_to(m)

m

dataset_area= dataset_new.drop(columns=['X5 latitude', 'X6 longitude'])
dataset_area