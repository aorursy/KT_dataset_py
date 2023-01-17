import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
df = pd.read_csv("../input/country_population.csv")
df.head()
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="Worldmap for countries population in 2016")
latitude = []
long = []
for i in df["Country Name"]:
    if i != None:
        location = geolocator.geocode(i)
        if location!=None:
            latitude.append(location.latitude)#, location.longitude)
            long.append(location.longitude)
        else:
            latitude.append(float("Nan"))#, location.longitude)
            long.append(float("Nan"))
    else:
        latitude.append(float("Nan"))#, location.longitude)
        long.append(float("Nan"))
df["Latitude"] = latitude
df.head()
df["Longitude"] = long
df.head()
type(df.iloc[1]["Longitude"])
df = df.dropna(axis=0)
import folium
world_map = folium.Map(location=[20, 0], tiles="Mapbox Bright", zoom_start=4)
for i in range(0,len(df)):
    folium.Marker([df.iloc[i]['Latitude'], df.iloc[i]['Longitude']], popup = "Population - " + str(df.iloc[i]['2016'])).add_to(world_map)
world_map
df[df["Country Code"] == "SVN"]
world_map.choropleth(
geo_data = df,
 name='choropleth',
 data=df,
 columns=['Country Name', '2016'],
 key_on='2016',
 fill_color='YlGn',
 fill_opacity=0.7,
 line_opacity=0.2,
 legend_name='Unemployment Rate (%)'
)