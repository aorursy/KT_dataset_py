# Running the below code to get geopandas and dependancies from Kaggle
!pip install -U -t /kaggle/working/ git+https://github.com/Kaggle/learntools.git@geospatial_edits
# Provide info about constants, functions and methods of the Python system
import sys
sys.path.append('/kaggle/working')
# Import all Libriaries to get set up
import math
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import MultiPolygon
from shapely.ops import nearest_points
from shapely.geometry import MultiPoint

import folium
from folium import Choropleth, Marker
from folium.plugins import HeatMap, MarkerCluster
# I will use the embed_map() function to visualize my maps.
def embed_map(m, file_name):
    from IPython.display import IFrame
    m.save(file_name)
    return IFrame(file_name, width='100%', height='500px')
collisions = gpd.read_file("../input/geospatial-learn-course-data/NYPD_Motor_Vehicle_Collisions/NYPD_Motor_Vehicle_Collisions/NYPD_Motor_Vehicle_Collisions.shp")
collisions.head()
collisions.shape
# Creating a heatmap capturing all the collisions
m_1 = folium.Map(location=[40.7, -74], zoom_start=11) 

# Visualize the collision data
HeatMap(data=collisions[['LATITUDE', 'LONGITUDE']], radius=9).add_to(m_1)

# Show the map
embed_map(m_1, "q_1.html")
# Loading the data about hospitals
hospitals = gpd.read_file("../input/geospatial-learn-course-data/nyu_2451_34494/nyu_2451_34494/nyu_2451_34494.shp")
hospitals.head()
# Create map called m_2 showing all hospitals locations
m_hospitals = folium.Map(location=[40.7, -74], zoom_start=11) 

# Add points to the map from all rows in the "hospitals" dataframe
for idx, row in hospitals.iterrows():
    Marker([row["latitude"], row["longitude"]],popup=row["name"], tooltip = row["capacity"]).add_to(m_hospitals)

# Show the map
embed_map(m_hospitals, "m_hospitals.html")
# Begin by creating a buffer of size 10000 around each point in hospitals.geometry. 10000 as it is in cms.
# Creating a buffer allows to map all collisions within 10k of a hospital.
coverage = gpd.GeoDataFrame(geometry=hospitals.geometry).buffer(10000)

# Then, use the unary_union attribute to create a MultiPolygon object, before checking to see if it contains each collision.
my_union = coverage.geometry.unary_union

# Creating new df capturing all collisions with crashes that occurred more than 10 kilometers from the closest hospital.
outside_range = collisions.loc[~collisions["geometry"].apply(lambda x: my_union.contains(x))]
# Calculating the % of collisions happening more than 10k away from the closest hospital
percentage = round(100*len(outside_range)/len(collisions), 2)
print ("There are",len(outside_range),"that occurred more than 10 kilometers from the closest hospital.")
print("Percentage of collisions more than 10 km away from the closest hospital: {}%".format(percentage))
def best_hospital(collision_location):
    idx_min = hospitals.geometry.distance(collision_location).idxmin()
    my_hospital = hospitals.iloc[idx_min]
    name = my_hospital["name"]
    latitude = my_hospital["latitude"]
    longitude = my_hospital["longitude"]
    return pd.Series({'name': name, 'lat': latitude, 'long': longitude})

# Print best hospital outside of range
print ("Best hospital outside of range")
print(best_hospital(outside_range.geometry.iloc[0]))
# Displaying the best out of range hospital in comparison to all outside_range collisions
# Choose the specific lat and lng wish to be mapped
lat_0 = 40.8481
long_0 = -73.8437

# Do not modify the code below this line
# Creating new df capturing lat_0 and lng_0
out_of_range_hosp = pd.DataFrame({'Latitude': [lat_0],'Longitude': [long_0]})
# Adding the coordonates and the EPSG2263 format
out_of_range_hosp_gdf = gpd.GeoDataFrame(out_of_range_hosp, geometry=gpd.points_from_xy
                                         (out_of_range_hosp.Longitude, out_of_range_hosp.Latitude))
out_of_range_hosp_gdf.crs = {'init' :'epsg:4326'}
out_of_range_hosp_gdf = out_of_range_hosp_gdf.to_crs(epsg=2263)

# make the map
m = folium.Map(location=[40.7, -74], zoom_start=10) 
for idx, row in out_of_range_hosp_gdf.iterrows():
    Marker([row['Latitude'], row['Longitude']]).add_to(m)
HeatMap(data=outside_range[['LATITUDE', 'LONGITUDE']], radius=9).add_to(m)

display(embed_map(m, 'q_6.html'))
highest_demand = outside_range.geometry.apply(best_hospital).name.value_counts().idxmax()
print("Information the highest demand hospital:")
print(hospitals.loc[hospitals['name'].isin([highest_demand])])
# Creating m-6
m_6 = folium.Map(location=[40.7, -74], zoom_start=11) 

# Adding the current hospitals location and coverage to m_6
folium.GeoJson(coverage.geometry.to_crs(epsg=4326)).add_to(m_6)
# Adding the heatmap capturing the "outside_range" data to m_6
HeatMap(data=outside_range[['LATITUDE', 'LONGITUDE']], radius=9).add_to(m_6)
# The following line is to get the Lat and Lng pop up on the map
folium.LatLngPopup().add_to(m_6)

# Show map
embed_map(m_6, 'm_6.html')
# Proposed location of hospital 1
lat_1 = 40.6714
long_1 = -73.8492

# Proposed location of hospital 2
lat_2 = 40.6702
long_2 = -73.7612

# Do not modify the code below this line
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
# make the map
m = folium.Map(location=[40.7, -74], zoom_start=11) 
folium.GeoJson(coverage.geometry.to_crs(epsg=4326)).add_to(m)
folium.GeoJson(new_coverage.geometry.to_crs(epsg=4326)).add_to(m)
for idx, row in new_gdf.iterrows():
    Marker([row['Latitude'], row['Longitude']]).add_to(m)
HeatMap(data=new_outside_range[['LATITUDE', 'LONGITUDE']], radius=9).add_to(m)
folium.LatLngPopup().add_to(m)
display(embed_map(m, 'q_6.html'))