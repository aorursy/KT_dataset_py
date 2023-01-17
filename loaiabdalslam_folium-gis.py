import numpy as np 
import pandas as pd 
df=pd.read_csv('../input/Public_Schools.csv')
df.head()
import folium 

lat=42.292193
lang=-71.097283
folium.Map(location=[lat,lang],zoom_start=12)
folium.Map(location=[lat,lang],zoom_start=8,tiles='Mapbox Control Room')
folium.Map(location=[lat,lang],zoom_start=12,tiles='Stamen Watercolor')
folium.Map(location=[lat,lang],zoom_start=12,tiles='Stamen Terrain')
folium.Map(location=[lat,lang],zoom_start=12,tiles='Stamen Toner')
colors={'ES':'red' , 
'K-8':'blue'  ,        
'HS':'purple'   ,        
'Special':'green',   
'6/7-12':'brick',    
'K-12':'red',       
'MS':'gray',   
'ELC':'orange',   
       }
boston_circle=folium.Map(location=[lat,lang],zoom_start=12)
for lat,lng,num in zip(df.Y,df.X,range(1,df.shape[0])): 
         popup = folium.Popup(df['SCH_NAME'][num], parse_html=True)
         folium.CircleMarker(
                    [lat,lng],
                    radius=6,
                    color=colors[df.iloc[num,-8]],
                    fill=True,
                    fill_color=colors[df.iloc[num,-8]],
                    fill_opacity=0.7,
                    popup=popup
            ).add_to(boston_circle)
boston_circle
boston_Marker=folium.Map(location=[lat,lang],zoom_start=12)
for lat,lng,num in zip(df.Y,df.X,range(1,df.shape[0])): 
         popup = folium.Popup(df['SCH_NAME'][num], parse_html=True)
         folium.Marker(
                    [lat,lng],
                    popup=popup,
                    icon=folium.Icon(color=colors[df.iloc[num,-8]])
            ).add_to(boston_Marker)
boston_Marker
boston_cluster=folium.Map(location=[lat,lang],zoom_start=12)
from folium import plugins
cluster=plugins.MarkerCluster().add_to(boston_cluster)

boston_Marker=folium.Map(location=[lat,lang],zoom_start=12)
for lat,lng,num in zip(df.Y,df.X,range(1,df.shape[0])): 
         popup = folium.Popup(df['SCH_NAME'][num], parse_html=True)
         folium.Marker(
                    [lat,lng],
                    popup=popup,
                    icon=folium.Icon(color=colors[df.iloc[num,-8]])
            ).add_to(cluster)
boston_cluster
boston_cluster.save('cluster.html')
