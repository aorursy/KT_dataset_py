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
def embed_map(m, file_name):

    from IPython.display import IFrame

    m.save(file_name)

    return IFrame(file_name, width='100%', height='500px')
collisions = gpd.read_file("../input/geospatial-learn-course-data/NYPD_Motor_Vehicle_Collisions/NYPD_Motor_Vehicle_Collisions/NYPD_Motor_Vehicle_Collisions.shp")

collisions.head()
m_1 = folium.Map(location=[40.7, -74], zoom_start=11) 



# Your code here: Visualize the collision data

#HeatMap(data=collisions[['LATITUDE', 'LONGITUDE']], radius=15).add_to(m_1)



mc = MarkerCluster()

for idx, row in collisions.iterrows():

    if not math.isnan(row['LONGITUDE']) and not math.isnan(row['LATITUDE']):

        mc.add_child(Marker([row['LATITUDE'], row['LONGITUDE']]))

m_1.add_child(mc)

    

# Uncomment to see a hint

#q_1.hint()



# Show the map

embed_map(m_1, "q_1.html")
# Get credit for your work after you have created a map

q_1.check()



# Uncomment to see our solution (your code may look different!)

#q_1.solution()
hospitals = gpd.read_file("../input/geospatial-learn-course-data/nyu_2451_34494/nyu_2451_34494/nyu_2451_34494.shp")

hospitals.head()
m_2 = folium.Map(location=[40.7, -74], zoom_start=11) 



# Your code here: Visualize the hospital locations

HeatMap(data=hospitals[['latitude', 'longitude']], radius=15).add_to(m_2)



# Uncomment to see a hint

#q_2.hint()

        

# Show the map

embed_map(m_2, "q_2.html")
# Get credit for your work after you have created a map

q_2.check()



# Uncomment to see our solution (your code may look different!)

#q_2.solution()
hosp = gpd.GeoDataFrame(geometry=hospitals.geometry)

hosp.head()

hosp_buffer = hosp.buffer(10000).geometry.unary_union
# Your code here

outside_range = collisions.loc[~collisions["geometry"].apply(lambda x: hosp_buffer.contains(x))]

# Check your answer

q_3.check()
# Lines below will give you a hint or solution code

#q_3.hint()

#q_3.solution()
percentage = round(100*len(outside_range)/len(collisions), 2)

print("Percentage of collisions more than 10 km away from the closest hospital: {}%".format(percentage))
def best_hospital(collision_location):

    # Your code here

    name = hospitals.iloc[hospitals.geometry.distance(collision_location).idxmin()]

    return name



# Test your function: this should suggest CALVARY HOSPITAL INC

print(best_hospital(outside_range.geometry.iloc[0]))



# Check your answer

q_4.check()
# Lines below will give you a hint or solution code

#q_4.hint()

#q_4.solution()
# Your code here

highest_demand = outside_range.geometry.apply(best_hospital).name.value_counts().idxmax()



# Check your answer

q_5.check()
# Lines below will give you a hint or solution code

#q_5.hint()

#q_5.solution()
m_6 = folium.Map(location=[40.7, -74], zoom_start=11) 



folium.GeoJson(coverage.geometry.to_crs(epsg=4326)).add_to(m_6)

HeatMap(data=outside_range[['LATITUDE', 'LONGITUDE']], radius=9).add_to(m_6)

folium.LatLngPopup().add_to(m_6)



embed_map(m_6, 'm_6.html')
# Your answer here: proposed location of hospital 1

lat_1 = 40.6827

long_1 = -73.7642



# Your answer here: proposed location of hospital 2

lat_2 = 40.6050

long_2 = -74.0058





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

    display(embed_map(m, 'q_6.html'))

except:

    q_6.hint()
# Uncomment to see one potential answer

#q_6.solution()