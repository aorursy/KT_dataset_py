# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
path="../input/alegeri-locale-bucuresti-2020-geocoded/PrimariaBucuresti.csv"

df= pd.read_csv(path)
df.head()
dif_column = -df['DAN NICUŞOR DANIEL-voturi'] + df["FIREA GABRIELA-PARTIDUL SOCIAL DEMOCRAT-voturi"]

df["diferenta"] = dif_column
from simple_folium import simple_folium

help(simple_folium)
df.columns
import folium
map_alegeri = folium.Map(location=[44.4174, 26.09],

                    zoom_start = 12) # Uses lat then lon. The bigger the zoom number, the closer in you get

map_alegeri# Calls the map to display
map_alegeri = folium.Map(location=[44.4174, 26.09],

                        tiles = "Stamen Toner",

                        zoom_start = 12)

map_alegeri
from folium import plugins

from folium.plugins import HeatMap
heatmapArr = df[['Latitude', 'Longitude','diferenta']].values

map_alegeri = folium.Map(location=[44.4174, 26.09],

                    zoom_start = 12) 



HeatMap(heatmapArr).add_to(map_alegeri)

#hm.add_to(map_alegeri)

# Display the map

map_alegeri
dif_column2 = df['DAN NICUŞOR DANIEL-voturi'] - df["FIREA GABRIELA-PARTIDUL SOCIAL DEMOCRAT-voturi"]

df["diferenta2"] = dif_column2
heatmapArr2 = df[['Latitude', 'Longitude','diferenta2']].values
map_alegeri = folium.Map(location=[44.4174, 26.09],

                    zoom_start = 12) 



HeatMap(heatmapArr2).add_to(map_alegeri)

#hm.add_to(map_alegeri)

# Display the map

map_alegeri
from folium import plugins
df.head()
map_hooray = folium.Map(location=[df.Latitude.median(), df.Longitude.median()],

                        tiles = "Stamen Toner",

                        zoom_start = 12)

#colormap = {0.0: 'pink', 0.3: 'blue', 0.5: 'green',  0.7: 'yellow', 1: 'red'}

HeatMap(heatmapArr2,min_opacity=0.1).add_to(map_hooray)

HeatMap(heatmapArr,min_opacity=0.1).add_to(map_hooray)







#hm.add_to(map_alegeri)

# Display the map



map_hooray
