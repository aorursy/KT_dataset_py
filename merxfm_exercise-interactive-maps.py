import pandas as pd

import geopandas as gpd



import folium

from folium import Choropleth

from folium.plugins import HeatMap



from learntools.core import binder

binder.bind(globals())

from learntools.geospatial.ex3 import *
plate_boundaries = gpd.read_file("../input/geospatial-learn-course-data/Plate_Boundaries/Plate_Boundaries/Plate_Boundaries.shp")

plate_boundaries['coordinates'] = plate_boundaries.apply(lambda x: [(b,a) for (a,b) in list(x.geometry.coords)], axis='columns')

plate_boundaries.drop('geometry', axis=1, inplace=True)



plate_boundaries.head()
# Load the data and print the first 5 rows

earthquakes = pd.read_csv("../input/geospatial-learn-course-data/earthquakes1970-2014.csv", parse_dates=["DateTime"])

earthquakes.head()
# Create a base map with plate boundaries

m_1 = folium.Map(location=[35,136], tiles='cartodbpositron', zoom_start=5)

for i in range(len(plate_boundaries)):

    folium.PolyLine(locations=plate_boundaries.coordinates.iloc[i], weight=2, color='black').add_to(m_1)



# Your code here: Add a heatmap to the map

HeatMap(data=earthquakes[['Latitude', 'Longitude']], radius=15).add_to(m_1)



# Show the map

m_1
# Get credit for your work after you have created a map

q_1.a.check()
# View the solution (Run this code cell to receive credit!)

q_1.b.solution()
# Create a base map with plate boundaries

m_2 = folium.Map(location=[35,136], tiles='cartodbpositron', zoom_start=5)

for i in range(len(plate_boundaries)):

    folium.PolyLine(locations=plate_boundaries.coordinates.iloc[i], weight=2, color='black').add_to(m_2)

    

# Your code here: Add a map to visualize earthquake depth



def colour(depth):

    if depth > 50: colour='orange';

    elif depth > 100: colour='red';

    else: colour='green'

    

    return colour



for index, earthquake in earthquakes.iterrows():

    folium.Circle(earthquake[['Latitude', 'Longitude']], radius=5000, color=colour(earthquake['Depth'])).add_to(m_2)



# View the map

m_2
# Get credit for your work after you have created a map

q_2.a.check()
# View the solution (Run this code cell to receive credit!)

q_2.b.solution()
# GeoDataFrame with prefecture boundaries

prefectures = gpd.read_file("../input/geospatial-learn-course-data/japan-prefecture-boundaries/japan-prefecture-boundaries/japan-prefecture-boundaries.shp")

prefectures.set_index('prefecture', inplace=True)

prefectures.head()
# DataFrame containing population of each prefecture

population = pd.read_csv("../input/geospatial-learn-course-data/japan-prefecture-population.csv")

population.set_index('prefecture', inplace=True)



# Calculate area (in square kilometers) of each prefecture

area_sqkm = pd.Series(prefectures.geometry.to_crs(epsg=32654).area / 10**6, name='area_sqkm')

stats = population.join(area_sqkm)



# Add density (per square kilometer) of each prefecture

stats['density'] = stats["population"] / stats["area_sqkm"]

stats.head()
# Create a base map

m_3 = folium.Map(location=[35,136], tiles='cartodbpositron', zoom_start=5)



# Your code here: create a choropleth map to visualize population density

Choropleth(geo_data=prefectures.__geo_interface__,

           key_on="feature.id",

           data=stats['density'],

           fill_color='YlOrRd',

           legend_name='Population density (per square kilometer)').add_to(m_3)



# View the map

m_3
# Get credit for your work after you have created a map

q_3.a.check()
# View the solution (Run this code cell to receive credit!)

q_3.b.solution()
# Create a base map

m_4 = folium.Map(location=[35,136], tiles='cartodbpositron', zoom_start=5)



# Your code here: create a map

Choropleth(geo_data=prefectures.__geo_interface__,

           key_on="feature.id",

           data=stats['density'],

           fill_color='BuPu',

           legend_name='Population density (per square kilometer)').add_to(m_4)



def colour(depth):

    if depth > 6.5: colour='red';

    else: colour='green'

    

    return colour



for index, earthquake in earthquakes.iterrows():

    folium.Circle(earthquake[['Latitude', 'Longitude']], 

                  radius=earthquake['Magnitude']**5.5, 

                  color=colour(earthquake['Magnitude']),

                  popup=("{} ({})").format(earthquake['Magnitude'], earthquake['DateTime'].year)).add_to(m_4)





# View the map

m_4
# Get credit for your work after you have created a map

q_4.a.check()
# View the solution (Run this code cell to receive credit!)

q_4.b.solution()