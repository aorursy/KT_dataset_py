import pandas as pd

import geopandas as gpd



import folium

from folium import Choropleth

from folium.plugins import HeatMap



from learntools.core import binder

binder.bind(globals())

from learntools.geospatial.ex3 import *
def embed_map(m, file_name):

    from IPython.display import IFrame

    m.save(file_name)

    return IFrame(file_name, width='100%', height='500px')
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



# Uncomment to see a hint

#q_1.a.hint()



# Show the map

embed_map(m_1, 'q_1.html')
# Get credit for your work after you have created a map

q_1.a.check()



# Uncomment to see our solution (your code may look different!)

#q_1.a.solution()
# View the solution

q_1.b.solution()
# Create a base map with plate boundaries

m_2 = folium.Map(location=[35,136], tiles='cartodbpositron', zoom_start=5)

for i in range(len(plate_boundaries)):

    folium.PolyLine(locations=plate_boundaries.coordinates.iloc[i], weight=2, color='black').add_to(m_2)

    

# # Your code here: Add a map to visualize earthquake depth

# HeatMap(data=earthquakes[['Latitude', 'Longitude']], radius=earthquakes['Depth']).add_to(m_2)



#Solution from exercise

# Custom function to assign a color to each circle

def color_producer(val):

    if val < 50:

        return 'forestgreen'

    elif val < 100:

        return 'darkorange'

    else:

        return 'darkred'



# Add a map to visualize earthquake depth

for i in range(0,len(earthquakes)):

    folium.Circle(

        location=[earthquakes.iloc[i]['Latitude'], earthquakes.iloc[i]['Longitude']],

        radius=2000,

        color=color_producer(earthquakes.iloc[i]['Depth'])).add_to(m_2)





# Uncomment to see a hint

#q_2.a.hint()



# View the map

embed_map(m_2, 'q_2.html')
# Get credit for your work after you have created a map

q_2.a.check()



# Uncomment to see our solution (your code may look different!)

q_2.a.solution()
# View the solution

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

# Add a choropleth map to the base map

Choropleth(geo_data=prefectures.__geo_interface__, 

           data=stats['density'], 

           key_on="feature.id", 

           fill_color='YlGnBu', 

           legend_name='Density of prefectures',

           bins = list(stats['density'].quantile([0, 0.25, 0.5,  0.75, 1]))

          ).add_to(m_3)



# Uncomment to see a hint

#q_3.a.hint()



# View the map

embed_map(m_3, 'q_3.html')
# Get credit for your work after you have created a map

q_3.a.check()



# Uncomment to see our solution (your code may look different!)

#q_3.a.solution()
# This is Tokaido city agglomeration



# View the solution

q_3.b.solution()
# Create a base map

m_4 = folium.Map(location=[35,136], tiles='cartodbpositron', zoom_start=5)



# Your code here: create a map



Choropleth(geo_data=prefectures.__geo_interface__, 

           data=stats['density'], 

           key_on="feature.id", 

           fill_color='YlGnBu', 

           legend_name='Density of prefectures',

           bins = list(stats['density'].quantile([0, 0.25, 0.5,  0.75, 1]))

          ).add_to(m_4)



def color_producer(val):

    if val < 7:

        return 'forestgreen'

    elif val < 8:

        return 'darkorange'

    else:

        return 'darkred'



# Add a map to visualize earthquake depth

for i in range(0,len(earthquakes)):

    folium.Circle(

        location=[earthquakes.iloc[i]['Latitude'], earthquakes.iloc[i]['Longitude']],

        radius=2000,

        color=color_producer(earthquakes.iloc[i]['Magnitude'])).add_to(m_4)





# Uncomment to see a hint

#q_4.a.hint()



# View the map

embed_map(m_4, 'q_4.html')
# Get credit for your work after you have created a map

q_4.a.check()



# Uncomment to see our solution (your code may look different!)

#q_4.a.solution()
# I think so this is Miyagi

# Because Miyage has a high level of density, a long sea coast and a large number of earthquakes. 

# Also, prefectures near the capital of Japan have good protection from earthquakes.



# View the solution

q_4.b.solution()