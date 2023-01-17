import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import geopandas as gpd
import math 

import contextily as ctx
df = pd.read_csv("../input/gymdata/gymdata.csv")
df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["long"], df["lat"]))
df.crs = {'init':'epsg:4326'}
ax = df.plot(figsize=(20,20), alpha=0.5, edgecolor="k")
ctx.add_basemap(ax, zoom=14,crs="epsg:4326", source=ctx.providers.CartoDB.Voyager)
ax.set_axis_off()
df.head()
df = df.rename(columns={"size ":"s"})
sz = []
for i in range(len(df)):
    sz.append(float(df.iloc[i]["s"] + 1))
for i in range(len(df)):
    if df.iloc[i]["s"] == 0:
        sz[i] += 1.5
import folium 
from folium import Choropleth, Marker
from folium.plugins import HeatMap

def embed_map(m, file_name):
    from IPython.display import IFrame
    m.save(file_name)
    return IFrame(file_name, width='100%', height='500px')

m_2 = folium.Map(location=[45.40,-75.69], zoom_start=12)

ott = gpd.read_file("../input/ottawa/ottawa.geojson")
pop = pd.DataFrame(ott)
pop = pop.sort_values(by=['POPEST'])

Choropleth(geo_data=ott,
           name="choropleth",
           data=pop,
           columns=["Name","POPEST"],
           fill_color='PuRd',
           key_on="feature.properties.Name",
           fill_opacity=0.5,
           line_opacity=0.2,
           legend_name="Population Estimate"
          ).add_to(m_2)

for i in range(0,len(df)):
    folium.Circle(
        location=[df.iloc[i]['lat'], df.iloc[i]['long']],
        tooltip=df.iloc[i]['name'],
        fill=True,
        radius=sz[i]*150).add_to(m_2)
    #Marker([df.iloc[i]['lat'], df.iloc[i]['long']], popup=df.iloc[i]['name']).add_to(m_2)
m_2
folium.Marker([45.410853, -75.692868],
       popup='<b>Location 1</b>',
       tooltip="Click").add_to(m_2)
folium.Marker([45.428000, -75.652771],
       popup='<b>Location 2</b>',
       tooltip="Click").add_to(m_2);
m_2
import matplotlib.pyplot as plt

with plt.xkcd():
    plt.scatter(df.numrating, df.rating)
    plt.xlabel("Number of ratings")
    plt.ylabel("Rating (/5)")
    plt.title("Rating vs Number of ratings")
fig2 = plt.figure();
with plt.xkcd():
    plt.scatter(sz, df.rating);
    plt.xlabel("size");
    plt.ylabel("Rating (/5)");
    plt.title("Rating vs size");
fig1 = plt.figure();
m_3 = folium.Map(location=[45.42,-75.70], zoom_start=12)

ott = gpd.read_file("../input/ottawa/ottawa.geojson")
pop = pd.DataFrame(ott)
pop = pop.sort_values(by=['POPEST'])

Choropleth(geo_data=ott,
           name="choropleth",
           data=pop,
           columns=["Name","POPEST"],
           fill_color='PuRd',
           key_on="feature.properties.Name",
           fill_opacity=0.5,
           line_opacity=0.2,
           legend_name="Population Estimate"
          ).add_to(m_3)

for i in range(0,len(df)):
    folium.Circle(
        location=[df.iloc[i]['lat'], df.iloc[i]['long']],
        tooltip=df.iloc[i]['name'],
        fill=True,
        radius=sz[i]*150).add_to(m_3)
    
folium.Marker([45.428000, -75.652771],
       popup='<b>Location 2</b>',
       tooltip="Click").add_to(m_3);
m_3