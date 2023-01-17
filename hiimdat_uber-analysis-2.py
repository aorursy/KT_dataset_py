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



df1 = pd.read_csv("/kaggle/input/uber-pickups-in-new-york-city/uber-raw-data-apr14.csv")

df2 = pd.read_csv("/kaggle/input/uber-pickups-in-new-york-city/uber-raw-data-aug14.csv")

df3 = pd.read_csv("/kaggle/input/uber-pickups-in-new-york-city/uber-raw-data-jul14.csv")

df4 = pd.read_csv("/kaggle/input/uber-pickups-in-new-york-city/uber-raw-data-jun14.csv")

df5 = pd.read_csv("/kaggle/input/uber-pickups-in-new-york-city/uber-raw-data-may14.csv")

df6 = pd.read_csv("/kaggle/input/uber-pickups-in-new-york-city/uber-raw-data-sep14.csv")
data_full = pd.concat([df1, df2, df3, df4, df5, df6])
data_full.shape
data_full.head()
data_full['Date/Time'] =  pd.to_datetime(data_full['Date/Time'])

data_full['Date'] =  pd.to_datetime(data_full['Date/Time'], format='%Y-%m-%d').dt.date

data_full['Hour'] =  pd.to_datetime(data_full['Date/Time'], format= '%H:%M').dt.hour

data_full['Minute'] =  pd.to_datetime(data_full['Date/Time'], format= '%H:%M').dt.minute

data_full['Weekday'] = data_full.Date.apply(lambda x: x.strftime('%A'))
data_full.head()
clus = data_full[['Lat', 'Lon']]

clus.dtypes
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from yellowbrick.cluster import KElbowVisualizer
model = KMeans()

visualizer = KElbowVisualizer(model, k = (1, 18))

visualizer.fit(clus)

visualizer.show()
kmeans = KMeans(n_clusters = 5, random_state = 0)

kmeans.fit(clus)
centroids = kmeans.cluster_centers_

centroids
clocation = pd.DataFrame(centroids, columns = ['Latitude', 'Longitude'])
clocation.head()
plt.scatter(clocation['Latitude'], clocation['Longitude'], marker = "x", color = 'R', s = 200)
import folium

centroid = clocation.values.tolist()



map = folium.Map(location = [40.71600413400166, -73.98971408426613], zoom_start = 10)

for point in range(0, len(centroid)):

    folium.Marker(centroid[point], popup = centroid[point]).add_to(map)



map
label = kmeans.labels_

label
data_new = data_full

data_new['Clusters'] = label

data_new
import seaborn as sb

sb.factorplot(data = data_new, x = "Clusters", kind = "count", size = 7, aspect = 2)
count_3 = 0

count_0 = 0

for value in data_new['Clusters']:

    if value == 3:

        count_3 += 1

    if value == 0:

        count_0 += 1

print(count_0, count_3)
new_location = [(40.86, -75.56)]

kmeans.predict(new_location)