import folium

#First install Folium in Anaconda

#conda install -c conda-forge folium=0.5.0 --yes
display(folium.Map())

import numpy as np

import pandas as pd
world_map = folium.Map(location=[56.130, -106.35], zoom_start=4)



# display world map

world_map



#coordinates of canada
# create a Stamen Toner map of the world centered around Canada

world_map = folium.Map(location=[56.130, -106.35], zoom_start=4, tiles='Stamen Toner')



# display map

world_map
# create a Stamen Toner map of the world centered around Canada

world_map = folium.Map(location=[56.130, -106.35], zoom_start=4, tiles='Stamen Terrain')



# display map

world_map
df = pd.read_csv('../input/dataset-pd/Police_Department_Incidents_-_Previous_Year__2016_.csv')
df.info()
df.head(2)
# get the first 100 crimes in the df_incidents dataframe

limit = 100

df_incidents = df.iloc[0:limit, :]
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

