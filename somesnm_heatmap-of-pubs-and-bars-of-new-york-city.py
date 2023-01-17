import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import folium

from folium import features

from folium.plugins import HeatMap

from folium.plugins import MarkerCluster



import seaborn
bars = pd.read_csv('../input/partynyc/bar_locations.csv')

bars.head()
bars.Borough.value_counts(ascending=True).tail().plot.barh();
pubs_map = folium.Map(location=[40.742459, -73.971765], zoom_start=12)

data = [[x[0], x[1], 1] for x in np.array(bars[['Latitude', 'Longitude']])]

HeatMap(data, radius = 20).add_to(pubs_map)

pubs_map
map_wb = folium.Map(location=[40.742459, -73.971765],zoom_start=12)#, tiles='Active nightlife zone')

mc = MarkerCluster()

for ind,row in bars.iterrows():

    mc.add_child(folium.CircleMarker(location=[row['Latitude'],row['Longitude']],

                        radius=1,color='#3185cc'))

map_wb.add_child(mc)

map_wb