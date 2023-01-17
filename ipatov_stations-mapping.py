from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from sklearn.cluster import KMeans



plt.style.use('ggplot')

fig = plt.figure(figsize=(8,8))

stations = pd.read_csv('../input/station.csv', index_col='station_id') 

trips = pd.read_csv('../input/trip.csv', skiprows=50793 )

bound = .001

to_station_id_counts = pd.DataFrame(trips['to_station_id'].value_counts())

from_station_id_counts = pd.DataFrame(trips['from_station_id'].value_counts())



stations = stations.join(to_station_id_counts).join(from_station_id_counts)

stations['traffic'] = stations.to_station_id + stations.from_station_id





right = stations.loc[:,['station_id','long','lat']]

df = trips.merge(right, left_on='to_station_id', right_on='station_id', how='outer')





statCoord = stations[['long','lat']]

clusters = KMeans(n_clusters=2, random_state=222).fit_predict(statCoord)

stations['clusters']=clusters



plt.hist(clusters, bins=2)
map = Basemap(projection='tmerc', 

            llcrnrlat=df['lat'].min()-bound, 

            urcrnrlat=df['lat'].max()+bound, 

            llcrnrlon=df['long'].min()-bound, 

            urcrnrlon=df['long'].max()+bound, 

            resolution='c', epsg=4269)

map.arcgisimage(service="NatGeo_World_Map", xpixels=1000, verbose=True) 



plt.title('Seattles Cycle Share System') 

map.scatter(stations.long, stations.lat, s=stations.traffic**(1/2.0), marker='o', c=clusters)

plt.show()