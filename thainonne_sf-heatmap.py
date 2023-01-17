# Heatmap for burglary

import pandas as pd

import matplotlib.pyplot as plt

import folium

from folium.plugins import HeatMap



df = pd.read_csv('../input/sanfranciso-crime-dataset/Police_Department_Incidents_-_Previous_Year__2016_.csv')



# X und Y not null

df = df[(df['X'] != 0) & (df['Y'] != 0)]



# Create dataset, crime = burglary

crime_burglary =  df[(df['Category'] == "BURGLARY")]

#testburglary = crime_burglary[:50]



# Create map, centre of San Francisco

map_van = folium.Map(location = [37.773972, -122.431297], zoom_start = 13)



# Create list with x- and y-coordinates, add to map, show map

heat_data = [[row['Y'],row['X']] for index, row in crime_burglary.iterrows()]

HeatMap(heat_data).add_to(map_van)



# folium.plugins.HeatMap(heat_data, radius = 10, min_opacity = 0.1, max_val = 100,gradient={.6: 'blue', .98: 'green', 1: 'red'}).add_to(map_van)



map_van



#testcrime_burglary = crime_burglary[:50]

#testcrime_burglary
# Heatmap for vehicle theft

import pandas as pd

import matplotlib.pyplot as plt

import folium

from folium.plugins import HeatMap



df = pd.read_csv('../input/sanfranciso-crime-dataset/Police_Department_Incidents_-_Previous_Year__2016_.csv')



# X und Y not null

df = df[(df['X'] != 0) & (df['Y'] != 0)]



# Create dataset, crime = burglary

crime_vehicle_theft =  df[(df['Category'] == "VEHICLE THEFT")]

#testvehth = vehth[:50]



# Create map, centre of San Francisco

map_van = folium.Map(location = [37.773972, -122.431297], zoom_start = 12)



# Create list with x- and y-coordinates, add to map, show map

heat_data = [[row['Y'],row['X']] for index, row in crime_vehicle_theft.iterrows()]

HeatMap(heat_data).add_to(map_van)



map_van
#testcrime_vehicle_theft = crime_vehicle_theft[:50]

#testcrime_vehicle_theft
# Heatmap for vandalism

import pandas as pd

import matplotlib.pyplot as plt

import folium

from folium.plugins import HeatMap



df = pd.read_csv('../input/sanfranciso-crime-dataset/Police_Department_Incidents_-_Previous_Year__2016_.csv')



# X und Y not null

df = df[(df['X'] != 0) & (df['Y'] != 0)]



# Create dataset, crime = burglary

crime_vandalism =  df[(df['Category'] == "VANDALISM")]

#testvehth = vehth[:50]



# Create map, centre of San Francisco

map_van = folium.Map(location = [37.773972, -122.431297], zoom_start = 12)



# Create list with x- and y-coordinates, add to map, show map

heat_data = [[row['Y'],row['X']] for index, row in crime_vandalism.iterrows()]

HeatMap(heat_data).add_to(map_van)



map_van
#testcrime_vandalism = crime_vandalism[:50]

#testcrime_vandalism