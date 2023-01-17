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
divisions = pd.read_csv('/kaggle/input/divisions.csv')

districts = pd.read_csv('/kaggle/input/districts.csv')

upazilas = pd.read_csv('/kaggle/input/upazilas.csv')
divisions.head(1)
m = folium.Map(

    location=[np.mean(divisions['lat']),np.mean(divisions['long'])],

    zoom_start=7

)

for index,rows in divisions.iterrows():

    if rows['lat'] and rows['long']:

        folium.Marker([rows['lat'], rows['long']], popup=rows['url']).add_to(m)

folium.LayerControl().add_to(m)

m.save('divisions.html')

m
districts.head(1)
m = folium.Map(

    location=[np.mean(districts['lat']),np.mean(districts['lon'])],

    zoom_start=7

)

for index,rows in districts.iterrows():

    if rows['lat'] and rows['lon']:

        folium.Marker([rows['lat'], rows['lon']], popup=rows['url']).add_to(m)

folium.LayerControl().add_to(m)

m.save('districts.html')

m
upazilas.head(1)
m = folium.Map(

    location=[np.mean(upazilas['lat']),np.mean(upazilas['lon'])],

    zoom_start=7

)

for index,rows in upazilas.iterrows():

    if rows['lat'] and rows['lon']:

        folium.Marker([rows['lat'], rows['lon']], popup=rows['url']).add_to(m)

folium.LayerControl().add_to(m)

m.save('upazilas.html')

m
m = folium.Map(

    location=[23.8196,90.4415],

    zoom_start=7

)

m.add_child(folium.LatLngPopup())

m.save('lat-long_miner.html')

m