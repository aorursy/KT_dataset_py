# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import geopandas as gpd
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# visualization
import matplotlib.pyplot as plt
import folium
import plotly.express as px
districts=gpd.read_file("/kaggle/input/india-district-wise-shape-files/output.shp")
#details1=gpd.read_file("/kaggle/input/india-district-wise-shape-files/output.prj")
districts.sample(5)
districts.columns.values
districts.info()
type(districts)
districts.describe()
#Creating a plot
districts.plot(figsize=(10,10))
districts.isna().sum()
# We can see all the null columns are quantitative so we can just put the median to fill na 

districts['distarea'].fillna(districts['distarea'].mean(),inplace=True)
districts['totalpopul'].fillna(districts['totalpopul'].mean(),inplace=True)
districts['totalhh'].fillna(districts['totalhh'].mean(),inplace=True)
districts['totpopmale'].fillna(districts['totpopmale'].mean(),inplace=True)
districts['totpopfema'].fillna(districts['totpopfema'].mean(),inplace=True)
states = districts.dissolve(by='statename',aggfunc='sum').reset_index()
states.head()
states.plot(figsize=(10,10),cmap='coolwarm',column='totalpopul',legend=True)
states[states['statename']=='Madhya Pradesh']
m=folium.Map(location=[23, 78.9629],tiles='cartodbpositron',min_zoom=4, max_zoom=8,zoom_start=5)
m
folium.Choropleth(states,#to get the Geo data
                  data=states,#data
                  key_on='feature.properties.statename', # feature.properties.key
                  columns=['statename', 'totalpopul'],   # [key, value]
                  fill_color='RdPu',                     # cmap
                  line_weight=0.15,                       # line wight (of the border)
                  line_opacity=0.5,                      # line opacity (of the border)
                  legend_name='Population').add_to(m)    # name on the legend color bar
    
    # add layer controls
folium.LayerControl().add_to(m)
m
states['female_sex_ratio']=(states['totpopfema']*1000)/states['totpopmale']
states.head()
folium.Choropleth(states,
                 data=states,
                 key_on='feature.properties.statename',
                 columns=['statename','female_sex_ratio'],
                 fill_color='PuBuGn',
                 line_weight=0.15,                       # line wight (of the border)
                 line_opacity=0.5,
                 legend_name='female_Sex_Ratio').add_to(m)

folium.LayerControl().add_to(m)