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
#import necessary packages

import geopandas as gpd
import folium
from folium import Choropleth, Circle, Marker
from folium.plugins import HeatMap, MarkerCluster
import math
#import traffic data from MMDA
traffic_data = pd.read_csv('../input/mmda-traffic-incident-data/data_mmda_traffic_spatial.csv')

#import manila map with boundaries
manila_map = gpd.read_file('../input/map-of-manila/Metropolitan Manila.shp', index_col=[0])
traffic_data.head()
#create subset of manila map where we only take the brgy, metro manila, and geometry
manila_map2 = manila_map.loc[:, ['NAME_3','NAME_2','geometry']].copy()
traffic_data_spots = gpd.GeoDataFrame(traffic_data, geometry=gpd.points_from_xy(traffic_data['Longitude'], traffic_data['Latitude']))
traffic_data_spots.crs = {'init': 'epsg:4326'}
#basic map
# ax = manila_map2.plot(figsize=(20,20), color='none', edgecolor='gainsboro', zorder=3)
# traffic_data_spots.plot(color='lightgreen', ax=ax)
manila_map2.head()
imap = folium.Map(location=[14.6091, 121.0223], tiles='openstreetmap', zoom_start=10)

mc = MarkerCluster()

for idx, row in traffic_data.iterrows():
    if not math.isnan(row['Longitude']) and not math.isnan(row['Latitude']):
        mc.add_child(Marker([row['Latitude'], row['Longitude']]))

imap.add_child(mc)
imap