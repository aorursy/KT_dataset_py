# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import geocoder
from geopy.geocoders import Nominatim
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# g = Nominatim('98 Edinburgh St, San Francisco, CA 94112, USA')
# g.latitude, g.longitude

employee = pd.read_csv("../input/employee-addresses/Employee_Addresses.csv")

employee.head()

employee['geocode'] = employee.loc[:, 'address'].map(lambda addr: geocoder.google(addr).latlng)

employee_sample = employee.loc[:, ['geocode', 'employee_id']]



    
import folium

from  folium.plugins  import MarkerCluster

# https://medium.com/@bobhaffner/folium-markerclusters-and-fastmarkerclusters-1e03b01cb7b1

map_sf = folium.Map(location=[37.76, -122.45],
                        zoom_start = 12)

mc = MarkerCluster()

for row in employee_sample.itertuples():
    mc.add_child(folium.Marker(location = row.geocode))

map_sf.add_child(mc)

display(map_sf)