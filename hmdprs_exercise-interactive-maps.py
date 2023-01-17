from learntools.core import binder

binder.bind(globals())

from learntools.geospatial.ex3 import *

print('Setup is completed!')



import pandas as pd

import geopandas as gpd



import folium

from folium import Choropleth

from folium.plugins import HeatMap
def embed_map(m, file_name):

    from IPython.display import IFrame

    m.save(file_name)

    return IFrame(file_name, width='100%', height='500px')
plate_boundaries = gpd.read_file("../input/geospatial-learn-course-data/Plate_Boundaries/Plate_Boundaries/Plate_Boundaries.shp")

plate_boundaries['coordinates'] = plate_boundaries.apply(lambda x: [(b,a) for (a,b) in list(x['geometry'].coords)], axis='columns')

plate_boundaries.drop('geometry', axis=1, inplace=True)



plate_boundaries.head()
# load data

earthquakes = pd.read_csv("../input/geospatial-learn-course-data/earthquakes1970-2014.csv", parse_dates=["DateTime"])

earthquakes.head()
# create a base map with plate boundaries

m_1 = folium.Map(location=[35,136], tiles='cartodbpositron', zoom_start=5)

for i in range(len(plate_boundaries)):

    folium.PolyLine(locations=plate_boundaries['coordinates'].iloc[i], weight=2, color='black').add_to(m_1)



# add a heatmap to the map

HeatMap(data=earthquakes[['Latitude', 'Longitude']], radius=10).add_to(m_1)



# uncomment to see a hint

# q_1.a.hint()



# show the map

embed_map(m_1, 'q_1.html')
# get credit for your work after you have created a map

q_1.a.check()



# uncomment to see our solution (your code may look different!)

# q_1.a.solution()
# View the solution (Run this code cell to receive credit!)

q_1.b.solution()
earthquakes.head()
# create a base map with plate boundaries

m_2 = folium.Map(location=[35,136], tiles='cartodbpositron', zoom_start=5)

for i in range(len(plate_boundaries)):

    folium.PolyLine(locations=plate_boundaries['coordinates'].iloc[i], weight=2, color='black').add_to(m_2)

    

# add a map to visualize earthquake depth

q1 = earthquakes['Depth'].quantile(q=.25)

q2 = earthquakes['Depth'].quantile(q=.5)

q3 = earthquakes['Depth'].quantile(q=.75)

def color_producer(val):

    if val < q1:

        return 'forestgreen'

    elif val < q2:

        return 'yellow'

    elif val < q3:

        return 'orange'

    else:

        return 'red'



# add a bubble map to the base map

for i in range(len(earthquakes)):

    folium.Circle(

        location=[earthquakes.iloc[i]['Latitude'], earthquakes.iloc[i]['Longitude']],

        radius=20,

        color=color_producer(earthquakes.iloc[i]['Depth'])

    ).add_to(m_2)



# uncomment to see a hint

# q_2.a.hint()



# view the map

embed_map(m_2, 'q_2.html')
# get credit for your work after you have created a map

q_2.a.check()



# uncomment to see our solution (your code may look different!)

# q_2.a.solution()
# View the solution (Run this code cell to receive credit!)

q_2.b.solution()
# create GeoDataFrame with prefecture boundaries

prefectures = gpd.read_file("../input/geospatial-learn-course-data/japan-prefecture-boundaries/japan-prefecture-boundaries/japan-prefecture-boundaries.shp")

prefectures.set_index('prefecture', inplace=True)

prefectures.head()
# create DataFrame containing population of each prefecture

population = pd.read_csv("../input/geospatial-learn-course-data/japan-prefecture-population.csv")

population.set_index('prefecture', inplace=True)



# calculate area (in square kilometers) of each prefecture

area_sqkm = pd.Series(prefectures['geometry'].to_crs(epsg=32654).area / 10**6, name='area_sqkm')

stats = population.join(area_sqkm)



# Add density (per square kilometer) of each prefecture

stats['density'] = stats["population"] / stats["area_sqkm"]

stats.head()
# create a base map

m_3 = folium.Map(location=[35,136], tiles='cartodbpositron', zoom_start=5)



# create a choropleth map to visualize population density

Choropleth(geo_data=prefectures.__geo_interface__, 

           data=stats['density'], 

           key_on="feature.id", 

           fill_color='YlGnBu', 

           legend_name='Population Density'

          ).add_to(m_3)



# uncomment to see a hint

# q_3.a.hint()



# view the map

embed_map(m_3, 'q_3.html')
# get credit for your work after you have created a map

q_3.a.check()



# uncomment to see our solution (your code may look different!)

# q_3.a.solution()
stats['density'].nlargest(3)
# view the solution (Run this code cell to receive credit!)

q_3.b.solution()
# create a base map

m_4 = folium.Map(location=[35,136], tiles='cartodbpositron', zoom_start=5)



# create a choropleth map to visualize population density

Choropleth(geo_data=prefectures.__geo_interface__, 

           data=stats['density'], 

           key_on="feature.id", 

           fill_color='YlGnBu', 

           legend_name='Population Density'

          ).add_to(m_4)



# a color producer function

q1 = earthquakes['Magnitude'].quantile(q=.25)

q2 = earthquakes['Magnitude'].quantile(q=.5)

q3 = earthquakes['Magnitude'].quantile(q=.75)

def color_producer(val):

    if val < q1:

        return 'forestgreen'

    elif val < q2:

        return 'yellow'

    elif val < q3:

        return 'orange'

    else:

        return 'red'



# add a bubble map

for i in range(len(earthquakes)):

    folium.Circle(

        location=[earthquakes.iloc[i]['Latitude'], earthquakes.iloc[i]['Longitude']],

        popup=(f"{earthquakes.iloc[i]['Magnitude']} ({earthquakes.iloc[i]['DateTime'].year})"),

        radius=20,

        color=color_producer(earthquakes.iloc[i]['Magnitude'])

    ).add_to(m_4)



# uncomment to see a hint

# q_4.a.hint()



# view the map

embed_map(m_4, 'q_4.html')
# get credit for your work after you have created a map

q_4.a.check()



# uncomment to see our solution (your code may look different!)

# q_4.a.solution()
# View the solution (Run this code cell to receive credit!)

q_4.b.solution()
# remove unnecessary file

# !rm /kaggle/working/q_5.html