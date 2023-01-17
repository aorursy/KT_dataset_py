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
data = pd.read_csv('/kaggle/input/covid19-tweets/covid19_tweets.csv')
data.head()
data['source_2']=data['source'].str.split(' ').str[-1]
data['source_2'].value_counts(normalize=True).head(10)
data_2=data[data['source_2']=='iPhone']

data_2=data_2[data_2['user_verified'].astype(str)=='True']

data_2.dropna(subset=['user_location'],inplace=True)

locations=pd.DataFrame(columns=['Location','Coordinates'])

locations['Location']=data_2['user_location'].unique()
from geopandas.tools import geocode

from geopy.geocoders import Nominatim

locator = Nominatim(user_agent='myGeocoder')

for i in range(locations.shape[0]):

    if locator.geocode(locations.loc[i]['Location'])!=None:

        locations.loc[i,'Coordinates'] = locator.geocode(locations.loc[i]['Location'])[1]

    else:

        locations.loc[i,'Coordinates'] = None
data_2=data_2.merge(locations,how='left',left_on='user_location',right_on='Location')

data_2.dropna(subset=['Coordinates'],inplace=True)

data_2.reset_index(inplace=True)
import folium

from folium.plugins import MarkerCluster

m = folium.Map(location=[0, 0],zoom_start=2)

mc = MarkerCluster()

for i in range(data_2.shape[0]):

    mc.add_child(folium.Marker(location=list(data_2.loc[i]['Coordinates']),popup = folium.Popup(data_2.loc[i]['user_name'])))

m.add_child(mc)
data_2['date_2'] = data_2['date'].str.split(' ').str.get(0)

for i in range(len(data_2['Coordinates'])):

    data_2.loc[i,'lat']=data_2.loc[i]['Coordinates'][0]

    data_2.loc[i,'lng']=data_2.loc[i]['Coordinates'][1]

dates = data_2['date'].str.split(' ').str.get(0).unique().tolist()
from folium.plugins import HeatMapWithTime

heat_data = [[[row['lat'],row['lng']] for index, row in data_2[data_2['date_2'] == i].iterrows()] for i in dates]

hm = HeatMapWithTime(data=heat_data, name=None, radius=7, min_opacity=0, max_opacity=0.8, 

                     scale_radius=False, gradient=None, use_local_extrema=False, auto_play=True, 

                     display_index=True, index_steps=1, min_speed=0.1, max_speed=10, speed_step=0.1, 

                     position='bottomleft', overlay=True, control=True, show=True)

data_2_tweets = folium.Map(tiles='OpenStreetMap', min_zoom=2) 

hm.add_to(data_2_tweets)

data_2_tweets
data_3=data[data['source_2']=='Android']

data_3=data_3[data_3['user_verified'].astype(str)=='True']

data_3.dropna(subset=['user_location'],inplace=True)

locations_android=pd.DataFrame(columns=['Location','Coordinates'])

locations_android['Location']=data_3['user_location'].unique()
from geopandas.tools import geocode

from geopy.geocoders import Nominatim

locator = Nominatim(user_agent='myGeocoder')

for i in range(locations_android.shape[0]):

    if locator.geocode(locations_android.loc[i]['Location'])!=None:

        locations_android.loc[i,'Coordinates'] = locator.geocode(locations_android.loc[i]['Location'])[1]

    else:

        locations_android.loc[i,'Coordinates'] = None
data_3=data_3.merge(locations_android,how='left',left_on='user_location',right_on='Location')

data_3.dropna(subset=['Coordinates'],inplace=True)

data_3.reset_index(inplace=True)
import folium

from folium.plugins import MarkerCluster

m = folium.Map(location=[0, 0],zoom_start=2)

mc = MarkerCluster()

for i in range(data_3.shape[0]):

    mc.add_child(folium.Marker(location=list(data_3.loc[i]['Coordinates']),popup = folium.Popup(data_3.loc[i]['user_name'])))

m.add_child(mc)
data_3['date_2'] = data_3['date'].str.split(' ').str.get(0)

for i in range(len(data_3['Coordinates'])):

    data_3.loc[i,'lat']=data_3.loc[i]['Coordinates'][0]

    data_3.loc[i,'lng']=data_3.loc[i]['Coordinates'][1]

dates = data_3['date'].str.split(' ').str.get(0).unique().tolist()
from folium.plugins import HeatMapWithTime

heat_data = [[[row['lat'],row['lng']] for index, row in data_3[data_3['date_2'] == i].iterrows()] for i in dates]

hm = HeatMapWithTime(data=heat_data, name=None, radius=7, min_opacity=0, max_opacity=0.8, 

                     scale_radius=False, gradient=None, use_local_extrema=False, auto_play=True, 

                     display_index=True, index_steps=1, min_speed=0.1, max_speed=10, speed_step=0.1, 

                     position='bottomleft', overlay=True, control=True, show=True)

data_3_tweets = folium.Map(tiles='OpenStreetMap', min_zoom=2) 

hm.add_to(data_3_tweets)

data_3_tweets
data_4=data[data['source_2']=='App']

data_4=data_4[data_4['user_verified'].astype(str)=='True']

data_4.dropna(subset=['user_location'],inplace=True)

locations_app=pd.DataFrame(columns=['Location','Coordinates'])

locations_app['Location']=data_4['user_location'].unique()
from geopandas.tools import geocode

from geopy.geocoders import Nominatim

locator = Nominatim(user_agent='myGeocoder')

for i in range(locations_app.shape[0]):

    if locator.geocode(locations_app.loc[i]['Location'])!=None:

        locations_app.loc[i,'Coordinates'] = locator.geocode(locations_app.loc[i]['Location'])[1]

    else:

        locations_app.loc[i,'Coordinates'] = None
data_4=data_4.merge(locations_app,how='left',left_on='user_location',right_on='Location')

data_4.dropna(subset=['Coordinates'],inplace=True)

data_4.reset_index(inplace=True)
import folium

from folium.plugins import MarkerCluster

m = folium.Map(location=[0, 0],zoom_start=2)

mc = MarkerCluster()

for i in range(data_4.shape[0]):

    mc.add_child(folium.Marker(location=list(data_4.loc[i]['Coordinates']),popup = folium.Popup(data_4.loc[i]['user_name'])))

m.add_child(mc)
data_4['date_2'] = data_4['date'].str.split(' ').str.get(0)

for i in range(len(data_4['Coordinates'])):

    data_4.loc[i,'lat']=data_4.loc[i]['Coordinates'][0]

    data_4.loc[i,'lng']=data_4.loc[i]['Coordinates'][1]

dates = data_4['date'].str.split(' ').str.get(0).unique().tolist()
from folium.plugins import HeatMapWithTime

heat_data = [[[row['lat'],row['lng']] for index, row in data_4[data_4['date_2'] == i].iterrows()] for i in dates]

hm = HeatMapWithTime(data=heat_data, name=None, radius=7, min_opacity=0, max_opacity=0.8, 

                     scale_radius=False, gradient=None, use_local_extrema=False, auto_play=True, 

                     display_index=True, index_steps=1, min_speed=0.1, max_speed=10, speed_step=0.1, 

                     position='bottomleft', overlay=True, control=True, show=True)

data_4_tweets = folium.Map(tiles='OpenStreetMap', min_zoom=2) 

hm.add_to(data_4_tweets)

data_4_tweets