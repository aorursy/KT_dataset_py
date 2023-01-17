# import the necessary libraries

import numpy as np 

import pandas as pd 



# Visualisation libraries

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()

from plotly.offline import init_notebook_mode, iplot 

import plotly.graph_objs as go

import plotly.offline as py

import pycountry

py.init_notebook_mode(connected=True)

import folium 

from folium import plugins



# Graphics in retina format 

%config InlineBackend.figure_format = 'retina' 



# Increase the default plot size and set the color scheme

plt.rcParams['figure.figsize'] = 8, 5

#plt.rcParams['image.cmap'] = 'viridis'



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Disable warnings 

import warnings

warnings.filterwarnings('ignore')
import geopandas as gpd

# Read in the data

location_data = gpd.read_file("../input/stack-overflow-user-location/top 1000 reputation location.csv")



# View the first five rows of the data

location_data.head()
# import plugins

from folium import Choropleth, Circle, Marker

from folium.plugins import HeatMap, MarkerCluster
# Load the data

user_locations = pd.read_csv("../input/stack-overflow-user-location/top 1000 reputation location.csv", encoding='latin-1')



# Drop rows with missing locations

user_locations.dropna(subset=['lat', 'lng'], inplace=True)



# Print the first five rows of the table

user_locations.head()
# how many rows left?

user_locations.info() # it's 824
import branca

import json

import numpy as np

#import vincent

from folium import IFrame

import numpy.ma as ma

import base64

# Create the map

mapp = folium.Map(location=[33.3473,120.1637], tiles='cartodbpositron', zoom_start=14) #this is my location.

# popup image(silicon valley) location

lat,lng = 37.3875, -122.0575

# popup

encoded = base64.b64encode(open('../input/pop-up-image-icon/silicon valley icon.png', 'rb').read())

html = '<img src="data:image/JPG;base64,{}">'.format

iframe = IFrame(html(encoded.decode("UTF-8")), width=256, height=256)

popup = folium.Popup(iframe, max_width=256)



icon = folium.Icon(color="Orange", icon="ok") # set the popup image icon

marker = folium.Marker(location=[lat, lng], popup=popup, icon=icon)

mapp.add_child(marker)



# Add points to the map

mc = MarkerCluster()

for idx, row in user_locations.iterrows():

        mc.add_child(Marker([row['lat'], row['lng']]))

mapp.add_child(mc)



#enable lat/lng popovers,help to find a location by interactively browsing the map

mapp.add_child(folium.LatLngPopup())



# Display the map

mapp



# mapp.save("MapImage.html")