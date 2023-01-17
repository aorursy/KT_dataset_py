# You can use the command below to install folium

# !pip install folium

# or

# !conda install -c conda-forge folium --yes 



# Note: this notebook was copied and that is solely because the original creator had the dataset already imported and I was too lazy to do that, 

#     all code written here are by me and you can check the original notebook for verification
# for visualizations

import numpy as np

import pandas as pd

import folium
# define the world map

world_map = folium.Map()



# display world map

world_map
# define Mexico's geolocation coordinates

mexico_latitude = 23.6345 

mexico_longitude = -102.5528



# define the world map centered around Mexico with a zoom level set to 4

mexico_map = folium.Map(location=[mexico_latitude, mexico_longitude], zoom_start=4)



# display Mexico's map

mexico_map
# define Mexico's geolocation coordinates

mexico_latitude = 23.6345 

mexico_longitude = -102.5528



# define the world map centered around Mexico with a higher zoom level

mexico_map = folium.Map(location=[mexico_latitude, mexico_longitude], zoom_start=6, tiles='Stamen Terrain')



# display mexico map

mexico_map
# reading the dataset

data = pd.read_csv('../input/Police_Department_Incidents_-_Previous_Year__2016_.csv')
# getting the shape of our dataset

data.shape
# get the first 100 crimes in the df_incidents dataframe

limit = 100

df_incidents = data.iloc[0:limit, :]
# San Francisco latitude and longitude values

latitude = 37.77

longitude = -122.42
# create map and display it

sanfran_map = folium.Map(location=[latitude, longitude], zoom_start=12)



# display the map of San Francisco

sanfran_map
# instantiate a feature group for the incidents in the dataframe

incidents = folium.map.FeatureGroup()



# loop through the 100 crimes and add each to the incidents feature group

for lat, lng, in zip(df_incidents.Y, df_incidents.X):

    incidents.add_child(

        folium.CircleMarker(

            [lat, lng],

            radius=5, # define how big you want the circle markers to be

            color='yellow',

            fill=True,

            fill_color='blue',

            fill_opacity=0.6

        )

    )



# add incidents to map

sanfran_map.add_child(incidents)
# instantiate a feature group for the incidents in the dataframe

incidents = folium.map.FeatureGroup()



# loop through the 100 crimes and add each to the incidents feature group

for lat, lng, in zip(df_incidents.Y, df_incidents.X):

    incidents.add_child(

        folium.CircleMarker(

            [lat, lng],

            radius=5, # define how big you want the circle markers to be

            color='yellow',

            fill=True,

            fill_color='blue',

            fill_opacity=0.6

        )

    )



# add pop-up text to each marker on the map

latitudes = list(df_incidents.Y)

longitudes = list(df_incidents.X)

labels = list(df_incidents.Category)



for lat, lng, label in zip(latitudes, longitudes, labels):

    folium.Marker([lat, lng], popup=label).add_to(sanfran_map)    

    

# add incidents to map

sanfran_map.add_child(incidents)
# create map and display it

sanfran_map = folium.Map(location=[latitude, longitude], zoom_start=12)



# loop through the 100 crimes and add each to the map

for lat, lng, label in zip(df_incidents.Y, df_incidents.X, df_incidents.Category):

    folium.CircleMarker(

        [lat, lng],

        radius=7, # define how big you want the circle markers to be

        color='yellow',

        fill=True,

        popup=label,

        fill_color='blue',

        fill_opacity=0.6

    ).add_to(sanfran_map)



# show map

sanfran_map
from folium import plugins



# let's start again with a clean copy of the map of San Francisco

sanfran_map = folium.Map(location = [latitude, longitude], zoom_start = 12)



# instantiate a mark cluster object for the incidents in the dataframe

incidents = plugins.MarkerCluster().add_to(sanfran_map)



# loop through the dataframe and add each data point to the mark cluster

for lat, lng, label, in zip(df_incidents.Y, df_incidents.X, df_incidents.Category):

    folium.Marker(

        location=[lat, lng],

        icon=None,

        popup=label,

    ).add_to(incidents)



# display map

sanfran_map