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
df_jakarta = pd.read_csv('../input/dki-jakarta/Pedagang-Lokasi-Sementara-.csv')
df_jakarta
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import folium
map_hooray = folium.Map(location=[-6.2088, 106.8456],
                        tiles = "Stamen Toner",
                        zoom_start = 12)
map_hooray
heat_df = df_jakarta[df_jakarta['Speed_limit']=='30'] # Reducing data size so it runs faster
heat_df = heat_df[heat_df['Year']=='2007'] # Reducing data size so it runs faster
from folium import plugins
from folium.plugins import HeatMap
import branca.colormap as cm
from collections import defaultdict
from folium.plugins import MarkerCluster

map_hooray = folium.Map(location=[-6.2088, 106.8456],
                        tiles = "Stamen Toner",
                    zoom_start = 11,
                       control_scale=True) 

# Ensure you're handing it floats
df_jakarta['latitude'] = df_jakarta['latitude'].astype(float)
df_jakarta['longitude'] = df_jakarta['longitude'].astype(float)

# Filter the DF for rows, then columns, then remove NaNs

heat_df = df_jakarta[['latitude', 'longitude']]
heat_df = heat_df.dropna(axis=0, subset=['latitude','longitude'])

#specify the min and max of my data
colormap = branca.colormap.linear.YlOrRd_09.scale(0,100)
colormap = colormap.to_step(index=[0,50,100,200])
colormap.caption = 'Jumlah UMKM'
colormap.add_to(map_hooray)

# List comprehension to make out list of lists
heat_data = [[row['latitude'],row['longitude']] for index, row in heat_df.iterrows()]

# Plot it on the map
HeatMap(heat_data).add_to(map_hooray)

# Display the map
map_hooray
df_jakarta.describe()
