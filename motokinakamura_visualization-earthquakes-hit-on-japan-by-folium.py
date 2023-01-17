#import libralies
import folium
import pandas as pd
from pandas import Series,DataFrame
import datetime
df = pd.read_csv("../input/SolarSystemAndEarthquakes.csv")
df.head()
df.dtypes
df.describe()
#change data types from object to datetime
df["earthquake.time"] = pd.to_datetime(df["earthquake.time"])
#pick up columns, [time, latitude, longitude, magnitude]
earth_quake_df = pd.DataFrame(columns=["time", "lat", "long", "mag","place"])

earth_quake_df["time"] = df["earthquake.time"]
earth_quake_df["lat"] = df["earthquake.latitude"]
earth_quake_df["long"] = df["earthquake.longitude"]
earth_quake_df["mag"] = df["earthquake.mag"]
earth_quake_df["place"] = df["earthquake.place"]
#select records which has "Japan"
eq_japan = earth_quake_df[earth_quake_df.place.str.contains("Japan")]

#reset index
eq_japan = eq_japan.reset_index(drop = True)

eq_japan.head()
eq_japan['year'] = eq_japan['time'].map(lambda x: x.year)
#set a new map
lat_mean = eq_japan.lat.mean()
long_mean = eq_japan.long.mean()

#set new map
map_jpn_marker = folium.Map(location=[lat_mean,long_mean] , zoom_start=5)
map_jpn_marker
#mark epicenters
for i in range(len(eq_japan)):
    index = i
    epicenter = [eq_japan.lat[index], eq_japan.long[index]]
    info = eq_japan.place[i] + ", yr = "+ str(eq_japan.year[i]) + ", mag = " + str(eq_japan.mag[i])
    folium.Marker(epicenter, popup=info, icon=folium.Icon(color="darkblue", icon = "flag")).add_to(map_jpn_marker)
    
map_jpn_marker
#set new map
map_jpn_circle = folium.Map(location=[lat_mean,long_mean] , zoom_start=5)
#put circle on epicenters
for i in range(0,len(eq_japan)):
    index = i
    info = eq_japan.place[i] + ", Y:"+ str(eq_japan.year[i]) + "," + "mag:"+ str(eq_japan.mag[i])
    folium.CircleMarker(
        location = [eq_japan.lat[index], eq_japan.long[index]],
        popup=info,
        radius=eq_japan.mag[i],
        color='#3186cc',
        fill_color='#3186cc'
    ).add_to(map_jpn_circle)
    

map_jpn_circle
eq_japan.year.unique()
#set colors for each years
colors_by_year = {
    #1980s
    1986:'darkred',
    1987:'darkred',
    1988:'darkred',
    1989:'darkred',
    #1990s
    1990:'darkblue',
    1991:'darkblue',
    1992:'darkblue',
    1993:'darkblue',
    1994:'darkblue',
    1995:'darkblue',
    1996:'darkblue',
    1997:'darkblue',
    1998:'darkblue',
    1999:'darkblue',
    #2000s
    2000:'darkgreen',
    2001:'darkgreen',
    2002:'darkgreen',
    2003:'darkgreen',
    2004:'darkgreen',
    2005:'darkgreen',
    2006:'darkgreen',
    2007:'darkgreen',
    2008:'darkgreen',
    2009:'darkgreen',
    #2010s
    2010:'gray',
    2011:'gray',
    2012:'gray',
    2013:'gray',
    2014:'gray',
    2015:'gray',
    2016:'gray'
}
#set new map
map_jpn_circle_year = folium.Map(location=[lat_mean,long_mean] , zoom_start=5)

#put circle on epicenters
for i in range(0,len(eq_japan)):
    info = eq_japan.place[i] + ", Y:"+ str(eq_japan.year[i]) + "," + "mag:"+ str(eq_japan.mag[i])
    folium.CircleMarker(
        location = [eq_japan.lat[i], eq_japan.long[i]],
        popup=info,
        radius=eq_japan.mag[i],
        color=colors_by_year[eq_japan.year[i]],
        fill_color=colors_by_year[eq_japan.year[i]]
    ).add_to(map_jpn_circle_year)
    

map_jpn_circle_year
