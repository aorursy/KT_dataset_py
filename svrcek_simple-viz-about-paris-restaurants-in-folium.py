# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebracode;



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Read the restaurants csv file

restaurants = pd.read_csv("/kaggle/input/restaurants-casvp.csv", sep=";")

# Inspect the first rows of restaurants

print(restaurants.head())

restaurants.info()
restaurants.shape
restaurants.head(15)
# Is there any null values? 

restaurants.isnull().sum()
# There are missing totally 4 values, for only one name of restaurant.

# We can release that restaurant name from the matrix: restaurants.shape (43, 6).
restaurants = restaurants.dropna()
# More readable address

restaurants['Address'] = restaurants['adresse']+ ', ' + restaurants['ville']# + ', '

restaurants.head()
del(restaurants['ville'])
del(restaurants['adresse'])
restaurants.Address[:5]
restaurants.head()
#restaurants["Nom restaurant"]#.head(20)
### Now almost each data are ready for Geo manipulation:
# This example uses the geopy module to produce latitude and longitudes.



from geopy.geocoders import Nominatim

from geopy.geocoders import ArcGIS

#from geopy.geocoders import ArcGIS

nom = ArcGIS()
restaurants['Coordinates'] = restaurants['Address'].apply(nom.geocode)

restaurants.head()
# I really don´t understand why the outcome is like that one above. In my local jupyter notebook is everything OK.

# Too other maps with the location/Coordinates will not be functional in this kernel (only locally!).
### Coordinates are ready and we can try this: 

restaurants.Coordinates[0]
restaurants.Coordinates[0].latitude
restaurants.Coordinates[0].longitude
restaurants.Coordinates[0:10]
# Below are all values written for an address.

restaurants.Coordinates.value_counts()[0:10]
restaurants["Latitude"] = restaurants["Coordinates"].apply(lambda x: x.latitude if x !=None else None)

restaurants["Longitude"] = restaurants["Coordinates"].apply(lambda x: x.longitude if x !=None else None)

restaurants.head()
# The last esthetic modification in the dataset.

del restaurants['tt']
restaurants.head()
import folium

#import branca

%matplotlib inline
# Example1:

# Map of Paris

map_r = folium.Map(location=[48.8647, 2.3490],

                        tiles = "Stamen Terrain",

                        zoom_start = 10)

map_r
# Example2:

#location = lat and long point to centre the map.

location = restaurants['Latitude'].mean(), restaurants['Longitude'].mean()



# Creating a basemap and the starting zoom.

map_r = folium.Map(location=[48.8647, 2.3490],zoom_start=12)



#Each location in the DataFrame is added as a marker.

for i in range(0,len(restaurants)):

        folium.Marker([restaurants['Latitude'].iloc[i],restaurants['Longitude'].iloc[i]]).add_to(map_r)

        

map_r
### Another Example with a pop_up when you click on the icon.

location = restaurants['Latitude'].mean(), restaurants['Longitude'].mean()

map_r = folium.Map(location=[48.8647, 2.3490],zoom_start=13)



for i in range(0,len(restaurants)):

       

    popup = folium.Popup('Nom restaurant', parse_html=True) 

    folium.Marker([restaurants['Latitude'].iloc[i],restaurants['Longitude'].iloc[i]],popup=popup).add_to(map_r)

map_r
# Restaurants types( only two)

restaurants.TYPE.value_counts()
location = restaurants['Latitude'].mean(), restaurants['Longitude'].mean()

map_r = folium.Map(location=[48.8647, 2.3490],zoom_start=13)



#The type for each restaurant  and the colour assigned to the basemap.

for i in range(0,len(restaurants)):

    type_of_rest = restaurants['TYPE'].iloc[i]

    if type_of_rest == 'E':

        color = 'blue'

    elif type_of_rest == 'S':

        color = 'red'

    

    popup = folium.Popup('Nom restaurant', parse_html=True) 

    folium.Marker([restaurants['Latitude'].iloc[i],restaurants['Longitude'].iloc[i]],popup=popup,icon=folium.Icon(color=color, icon='info-sign')).add_to(map_r)



map_r
# A setting for the last showcase of the map with the name and address of the restaurants in Paris.



def res_html(row):

    i = row

    

    Name_of_Restaurant = restaurants['Nom restaurant'].iloc[i]       

    Address = restaurants['Address'].iloc[i]

    

    left_color = "#2A799C"

    right_color = "#C5DCE7"

    

    html = """<!DOCTYPE html>

<html>



<head>

<h4 style="margin-bottom:0"; width="300px">{}</h4>""".format(Name_of_Restaurant) + """



</head>

    <table style="height: 126px; width: 300px;">

<tbody>

<tr>

<td style="background-color: """+ left_color +""";"><span style="color: #ffffff;">Address</span></td>

<td style="width: 200px;background-color: """+ right_color +""";">{}</td>""".format(Address) + """

</tr>



</tbody>

</table>

</html>

"""

    return html
location = restaurants['Latitude'].mean(), restaurants['Longitude'].mean()

map_r = folium.Map(location=[48.8647, 2.3490],zoom_start=14,min_zoom=5)



for i in range(0,len(restaurants)):

    html = res_html(i)

 

    iframe = branca.element.IFrame(html=html,width=320,height=150)

    popup = folium.Popup(iframe,parse_html=True)

    

    folium.Marker([restaurants['Latitude'].iloc[i],restaurants['Longitude'].iloc[i]],

                  popup=popup,icon=folium.Icon(color=color, icon='info-sign')).add_to(map_r)



map_r
# For the viz Using GeoPandas you need to download the files with all shp, shx,...files. 

# Everything what you need is to search these files simply on Google.
# Visualisation Using GeoPandas & MatPlotLib

import matplotlib.pyplot as plt

import descartes

import geopandas as gpd

from shapely.geometry import Point, Polygon



%matplotlib inline
### In GeoPandas again, you need to import a .shp file to plot on. ... Use search on Google.

state_map = gpd.read_file('../france-places-shape/places.shp')

fig,ax = plt.subplots(figsize = (15,15))

state_map.plot(ax = ax)