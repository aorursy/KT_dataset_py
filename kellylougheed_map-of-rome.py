# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import folium



styles = ["Stamen Terrain", "Stamen Toner", "Mapbox Bright"]



map_rome = folium.Map(location=[41.9028, 12.4964], # latitude, longitude

                      # tiles="Stamen Terrain",

                      zoom_start = 14) # The bigger the zoom number, the closer in you get

map_rome
# Get latitude & longitude from addresses: https://www.latlong.net/convert-address-to-lat-long.html



folium.Marker([41.8925, 12.4853], 

              popup="Roman Forum", 

             ).add_to(map_rome)



folium.Marker([41.896580, 12.474710], 

              icon=folium.Icon(color='green', icon='star', prefix='fa'),

              popup="Buddy Italian Restaurant Cafe").add_to(map_rome)



# Use icon names from https://fontawesome.com/

# Note: not all icons work



# To find all the options for Folium icons, run: help(folium.Icon)



map_rome
help(folium.Icon)