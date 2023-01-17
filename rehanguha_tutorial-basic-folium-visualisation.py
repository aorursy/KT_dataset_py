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
import pandas as pd

import numpy as np

import folium

from folium.plugins import HeatMap
df = pd.read_csv("/kaggle/input/us-accidents/US_Accidents_May19.csv")
df.shape
df.info()
df.head(3).T
def plot_map1(LatLong, city=None):

    accident_map = folium.Map(location=LatLong, 

                           tiles = "Stamen Toner",

                           zoom_start = 10)

    if city != None:

        data_heatmap = df[df["City"] == city]

    else:

        data_heatmap = df.copy()

    data_heatmap = data_heatmap[['Start_Lat','Start_Lng']]

    data_heatmap = [[row['Start_Lat'],row['Start_Lng']] for index, row in data_heatmap.iterrows()]

    HeatMap(data_heatmap, radius=10).add_to(accident_map)

    return accident_map
df[df['City']=='New York'].shape
plot_map1([40.712776,-74.005974], city='New York')
df.City.values
def plot_map2(city):

    data_heatmap = df[df["City"] == city]

    lat = data_heatmap['Start_Lat'].iloc[0]

    long = data_heatmap['Start_Lng'].iloc[0]

    LatLong = [lat, long]

    accident_map = folium.Map(location=LatLong, 

                           tiles = "cartodbpositron",

                           zoom_start = 10)



    data_heatmap = data_heatmap[['Start_Lat','Start_Lng']]

    data_heatmap = [[row['Start_Lat'],row['Start_Lng']] for index, row in data_heatmap.iterrows()]

    HeatMap(data_heatmap, radius=10).add_to(accident_map)

    return accident_map
plot_map2(city = "Sylmar")