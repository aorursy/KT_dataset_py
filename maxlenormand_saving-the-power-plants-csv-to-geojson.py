import numpy as np

import os



import pandas as pd



# Plotting libraries

import matplotlib.pyplot as plt
# Geospatial libraries that we will be using for this

import folium

import geopandas as gpd

from shapely.geometry import Point
power_plants = pd.read_csv('/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gppd/gppd_120_pr.csv')

power_plants.columns
# Plotting the .geo info of the first power plant

power_plants['.geo'][0]
longitude = float(power_plants['.geo'][0].split("[")[1][:-2].split(",")[0])

latitude = float(power_plants['.geo'][0].split("[")[1][:-2].split(",")[1])



print(f'Longitude: {longitude}, lattitude: {latitude}')
# We can create a new column containing this information

power_plants['longitude'] = [float(power_plants['.geo'][point].split("[")[1][:-2].split(",")[0]) for point in range(power_plants.shape[0])]

power_plants['latitude'] = [float(power_plants['.geo'][point].split("[")[1][:-2].split(",")[1]) for point in range(power_plants.shape[0])]
geometry_power_plants = [Point(x,y) for x,y in zip(power_plants.longitude, power_plants.latitude)]
del power_plants['.geo']
gdf_power_plants = gpd.GeoDataFrame(power_plants, crs = {'init': 'epsg: 4326'}, geometry = geometry_power_plants)

gdf_power_plants.head()
gdf_power_plants.plot()
# Plot on the map

lat=18.200178; lon=-66.664513

plot = folium.Map(location = (lat, lon), zoom_start=9)



for plant_lat, plant_long in zip(gdf_power_plants.latitude, gdf_power_plants.longitude):

    folium.Marker(location = (plant_lat, plant_long),

                    icon = folium.Icon(color='blue')).add_to(plot)

    

plot
# Saving the geodataframe for easy use later

gdf_power_plants.to_file('Geolocated_gppd_120_pr.geojson', driver='GeoJSON')