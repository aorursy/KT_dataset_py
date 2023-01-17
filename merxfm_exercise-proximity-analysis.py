import math

import geopandas as gpd

import pandas as pd

from shapely.geometry import MultiPolygon



import folium

from folium import Choropleth, Marker

from folium.plugins import HeatMap, MarkerCluster



from learntools.core import binder

binder.bind(globals())

from learntools.geospatial.ex5 import *
collisions = gpd.read_file("../input/geospatial-learn-course-data/NYPD_Motor_Vehicle_Collisions/NYPD_Motor_Vehicle_Collisions/NYPD_Motor_Vehicle_Collisions.shp")

collisions.head()
m_1 = folium.Map(location=[40.7, -74], zoom_start=11) 



# Your code here: Visualize the collision data

HeatMap(data=collisions[['LATITUDE', 'LONGITUDE']], radius=9).add_to(m_1)



# Show the map

m_1
# Get credit for your work after you have created a map

q_1.check()
hospitals = gpd.read_file("../input/geospatial-learn-course-data/nyu_2451_34494/nyu_2451_34494/nyu_2451_34494.shp")

hospitals.head()
m_2 = folium.Map(location=[40.7, -74], zoom_start=11) 



# Your code here: Visualize the hospital locations

for index, row in hospitals.iterrows():

    Marker(row[['latitude', 'longitude']], popup=row['name']).add_to(m_2)



# Uncomment to see a hint

#q_2.hint()

        

# Show the map

m_2
# Get credit for your work after you have created a map

q_2.check()
# Your code here

ten_km_buffer = hospitals.geometry.buffer(10e3)

ten_km_union = ten_km_buffer.unary_union



mask = collisions.geometry.within(ten_km_union)



outside_range = collisions[~mask]



display(outside_range.head())



# Check your answer

q_3.check()
percentage = round(100*len(outside_range)/len(collisions), 2)

print("Percentage of collisions more than 10 km away from the closest hospital: {}%".format(percentage))
def best_hospital(collision_location):

    # Your code here

    closest_hospital_index = hospitals.geometry.distance(collision_location).idxmin()

    name = hospitals.loc[closest_hospital_index, 'name']

    return name



# Test your function: this should suggest CALVARY HOSPITAL INC

print(best_hospital(outside_range.geometry.iloc[0]))



# Check your answer

q_4.check()
# Your code here

highest_demand = outside_range.geometry.apply(best_hospital).value_counts().nlargest(1).index[0]



display(highest_demand)



# Check your answer

q_5.check()
# Lines below will give you a hint or solution code

q_5.hint()

q_5.solution()
m_6 = folium.Map(location=[40.7, -74], zoom_start=11) 



coverage = gpd.GeoDataFrame(geometry=hospitals.geometry).buffer(10000)

folium.GeoJson(coverage.geometry.to_crs(epsg=4326)).add_to(m_6)

HeatMap(data=outside_range[['LATITUDE', 'LONGITUDE']], radius=9).add_to(m_6)

folium.LatLngPopup().add_to(m_6)



m_6
# Your answer here: proposed location of hospital 1

lat_1 = 40.6796

long_1 = -73.8673



# Your answer here: proposed location of hospital 2

lat_2 = 40.6770

long_2 =  -73.7560





# Do not modify the code below this line

try:

    new_df = pd.DataFrame(

        {'Latitude': [lat_1, lat_2],

         'Longitude': [long_1, long_2]})

    new_gdf = gpd.GeoDataFrame(new_df, geometry=gpd.points_from_xy(new_df.Longitude, new_df.Latitude))

    new_gdf.crs = {'init' :'epsg:4326'}

    new_gdf = new_gdf.to_crs(epsg=2263)

    # get new percentage

    new_coverage = gpd.GeoDataFrame(geometry=new_gdf.geometry).buffer(10000)

    new_my_union = new_coverage.geometry.unary_union

    new_outside_range = outside_range.loc[~outside_range["geometry"].apply(lambda x: new_my_union.contains(x))]

    new_percentage = round(100*len(new_outside_range)/len(collisions), 2)

    print("(NEW) Percentage of collisions more than 10 km away from the closest hospital: {}%".format(new_percentage))

    # Did you help the city to meet its goal?

    q_6.check()

    # make the map

    m = folium.Map(location=[40.7, -74], zoom_start=11) 

    folium.GeoJson(coverage.geometry.to_crs(epsg=4326)).add_to(m)

    folium.GeoJson(new_coverage.geometry.to_crs(epsg=4326)).add_to(m)

    for idx, row in new_gdf.iterrows():

        Marker([row['Latitude'], row['Longitude']]).add_to(m)

    HeatMap(data=new_outside_range[['LATITUDE', 'LONGITUDE']], radius=9).add_to(m)

    folium.LatLngPopup().add_to(m)

    display(m)

except:

    q_6.hint()