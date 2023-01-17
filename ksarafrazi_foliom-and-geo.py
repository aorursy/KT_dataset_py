import folium
import pandas as pd
from shapely.geometry import Point, Polygon
import numpy as np
df = pd.read_csv('../input/track.csv', parse_dates=['time'])
df.index = df['time']

df.head()
#Creating a new map
m = folium.Map(location=[df['lng'].mean(), df['lat'].mean()], zoom_start=15)

#Adding a marker
row = df.iloc[321]
marker = folium.CircleMarker([row['lng'],row['lat']], radius = 5, color = 'red', popup = 'Hi')
marker.add_to(m)

m
m = folium.Map(location=[df['lng'].mean(), df['lat'].mean()], zoom_start=15)
mdf = df.resample('T').mean()
#Add the whole track to the map
def add_marker(row):
    marker = folium.CircleMarker([row['lng'], row['lat']], radius=5, color='red', popup=row.name.strftime('%H:%M'))
    marker.add_to(m)

mdf.apply(add_marker, axis=1)
m
#Creating a column of points
mdf['pt'] = mdf[['lng', 'lat']].apply(Point, axis=1)
mdf.head()
#Creating a polygon 
mean_lng, max_lng = mdf['lng'].mean(), mdf['lng'].max()
mean_lat, max_lat = mdf['lat'].mean(), mdf['lat'].max()

poly = Polygon([
    [mean_lng, mean_lat],
    [mean_lng, max_lat],
    [max_lng, max_lat],
    [max_lng, mean_lat]
])

#Plotting the polygon the map
m = folium.Map(location=[df['lng'].mean(), df['lat'].mean()], zoom_start=15)
mdf = df.resample('T').mean()
mdf['pt'] = mdf[['lng', 'lat']].apply(Point, axis=1)

#Coloring the points inside our polygon red, and other points green
def add_marker(row):
    color = 'red' if poly.intersects(row['pt']) else 'green'
    marker = folium.CircleMarker([row['lng'], row['lat']], radius=5, color = color, popup=row.name.strftime('%H:%M'),
                                fill_color=color)
    marker.add_to(m)
m.add_child(folium.PolyLine(np.stack(poly.exterior.xy).T, color='yellow'))
mdf.apply(add_marker, axis=1)
m
