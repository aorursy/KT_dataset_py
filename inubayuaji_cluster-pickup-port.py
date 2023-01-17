# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import folium # plot map

from folium.plugins import MarkerCluster



import matplotlib.pyplot as plt # plot

%matplotlib inline



from sklearn.cluster import KMeans # model



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# read data

data = pd.read_csv('/kaggle/input/taxi-routes-for-mexico-city-and-quito/mex_clean.csv')

data.head()
# calculate vendor and how much pickup activities

data['vendor_id'].value_counts()
# chose one vendor and plot it

taxi_libre = data[data['vendor_id'] == 'México DF Taxi Libre']



plt.figure(figsize=(12, 6))

plt.scatter(taxi_libre['pickup_latitude'], taxi_libre['pickup_longitude'])

plt.show()
# use kmeans with centeroid random to get ceteroid of found cluster

kmeans = KMeans(n_clusters=3, random_state=10)



taxi_libre['cluster'] = kmeans.fit_predict(taxi_libre[['pickup_latitude', 'pickup_longitude']])
# plot initial cluster

plt.figure(figsize=(12, 6))

plt.scatter(taxi_libre['pickup_latitude'][taxi_libre['cluster'] == 0], taxi_libre['pickup_longitude'][taxi_libre['cluster'] == 0], c='r', label='c0')

plt.scatter(taxi_libre['pickup_latitude'][taxi_libre['cluster'] == 1], taxi_libre['pickup_longitude'][taxi_libre['cluster'] == 1], c='y', label='c1')

plt.scatter(taxi_libre['pickup_latitude'][taxi_libre['cluster'] == 2], taxi_libre['pickup_longitude'][taxi_libre['cluster'] == 2], c='g', label='c2')

plt.legend()

plt.show()
c0_count = taxi_libre[taxi_libre['cluster'] == 0].shape[0]

c1_count = taxi_libre[taxi_libre['cluster'] == 1].shape[0]

c2_count = taxi_libre[taxi_libre['cluster'] == 2].shape[0]

print('cluster0: {}\ncluster1: {}\ncluster2: {}'.format(c0_count, c1_count, c2_count))
# drop cluster 1 and 2

taxi_libre0 = taxi_libre[taxi_libre['cluster'] == 0]

taxi_libre0.drop(['cluster'], axis=1, inplace=True)

taxi_libre0.head()
taxi_libre0['cluster'] = kmeans.fit_predict(taxi_libre0[['pickup_latitude', 'pickup_longitude']])

taxi_libre0.head()
# where centeroid of cluster

centeroid = kmeans.cluster_centers_

plt.figure(figsize=(12, 6))

plt.scatter(taxi_libre0['pickup_latitude'][taxi_libre0['cluster'] == 0], taxi_libre0['pickup_longitude'][taxi_libre0['cluster'] == 0], c='r', label='c0')

plt.scatter(taxi_libre0['pickup_latitude'][taxi_libre0['cluster'] == 1], taxi_libre0['pickup_longitude'][taxi_libre0['cluster'] == 1], c='y', label='c1')

plt.scatter(taxi_libre0['pickup_latitude'][taxi_libre0['cluster'] == 2], taxi_libre0['pickup_longitude'][taxi_libre0['cluster'] == 2], c='g', label='c2')

plt.scatter(centeroid[:,0], centeroid[:,1], c='b', marker='d', label='centeroid')

plt.legend()

plt.show()
# show port location in map

m = folium.Map(

    location=[centeroid[0][0], centeroid[0][1]],

    set_zoom=17,

)



for i in range(centeroid.shape[0]):

    folium.Marker([centeroid[i][0], centeroid[i][1]], poop=str(i)).add_to(m)



m