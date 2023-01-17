# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
"""

This dataset is the average number of daily passenger for each stations in Yamanote Line in Japan.

"""

base_df=pd.read_csv("../input/station_location_dailypassenger_yamanote.csv")

base_df
plt.bar(base_df["name"],base_df["passenger"])

plt.ylabel("The number of Passenger")

plt.xlabel("Station")

plt.xticks(rotation=90,size='small')
!pip install folium
import folium
"""

In this section, I ploted the number of passenger for each station in Tokyo on the map.

I asserted the initial condition of the map in the function of the Map.

"""

tokyo_map = folium.Map(zoom_start=12,width=500,height=500,location=[35.677730,139.734813])

for i, r in base_df.iterrows():

    #setting for the popup

    popup=folium.Popup(r['name'],max_width=1000)

    #Plotting the Marker for each stationsト

    folium.map.Marker(

        location=[r['lat'], r['lon']], 

        popup=popup,

        icon=folium.Icon(color="green",icon="train", prefix='fa')

    ).add_to(tokyo_map)

    

    folium.vector_layers.CircleMarker(

        location=[r['lat'], r['lon']], 

        radius=r['passenger']/10000,

        color='#3186cc',

        fill_color='#3186cc'

    ).add_to(tokyo_map)
tokyo_map