# !pip install calmap
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px

import matplotlib.pyplot as plt

import matplotlib

import matplotlib.dates as mdates

import folium 

#import calmap

# Input data files

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('/kaggle/input/apr21-world-corona/Corona_Country _Data .csv')

data=df.copy()

data.drop(0)

world_map=pd.DataFrame({

    

    'Country':list(data['Country']),

    'Total Cases' :list(data['Total Cases']),

    'Deaths':list(data['Deaths']),

    'Recovered':list(data['Recovered']),

    'Active':list(data['Active']),

    'lat':list(data['lat']),

    'long':list(data['long'])    

})
_map = folium.Map(location=[23,80], tiles="Stamen Toner", zoom_start=4)



for lat, lon, value,value1,value2,value3, name in zip(world_map['lat'], world_map['long'], world_map['Total Cases'],world_map['Deaths'],world_map['Recovered'],world_map['Active'], world_map['Country']):

    folium.CircleMarker([lat, lon],

                        radius= (int((np.log(value+1.00001))))*3,

                        popup = ('<strong>Country</strong>: ' + str(name).capitalize() + '<br>'

                                '<strong>Confirmed Cases</strong>: ' + str(value) + '<br>'

                                '<strong>Death Cases</strong>: ' + str(value1) + '<br>'

                                '<strong>Recovered Cases</strong>: ' + str(value2) + '<br>'

                                '<strong>Active Cases</strong>: ' + str(value3) + '<br>'),

                        color='#ff6600',

                        

                        fill_color='#ff8533',

                        fill_opacity=0.8 ).add_to(_map)

_map