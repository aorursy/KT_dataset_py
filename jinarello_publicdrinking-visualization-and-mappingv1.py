import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import math

import geopandas as gpd

import pandas as pd

from shapely.geometry import MultiPolygon



import folium

from folium import Choropleth, Circle, Marker

from folium import plugins

from folium.plugins import HeatMap, MarkerCluster

from folium.plugins import FloatImage

# from learntools.core import binder

# binder.bind(globals())

# from learntools.geospatial.ex5 import *



def embed_map(m, file_name):

    from IPython.display import IFrame

    m.save(file_name)

    return IFrame(file_name, width='100%', height='500px')
gdata = pd.read_csv("../input/nyc-public-drinking-infractions/Drinking_In_Public.csv")

gdata.head()
gdata = gdata.drop(['Vehicle Type', 'Taxi Company Borough', 'Taxi Pick Up Location', 'Bridge Highway Name', 'Bridge Highway Direction', 'Road Ramp', 'Bridge Highway Segment'], axis=1)
gdata["Location Type"] = gdata["Location Type"].fillna(0, axis=0)

gdata = gdata.dropna(subset=['Longitude'])

gdata = gdata.dropna(subset=['Latitude'])

nans = lambda df: df[df.isnull().any(axis=1)]

nans(gdata)
m_1 = folium.Map(location=[40.7, -74], zoom_start=11) 

HeatMap(data=gdata[['Latitude', 'Longitude']], radius=8).add_to(m_1)

display(m_1)
m_2 = folium.Map(location=[40.7, -74], tiles='cartodbpositron', zoom_start=11) 



# for i in range(0,len(gdata)):

#     Circle(

#         location=[gdata.iloc[i]['Latitude'], gdata.iloc[i]['Longitude']],

#         radius=20,

#         color='forestgreen').add_to(m_2)



mc = MarkerCluster()

for idx, row in gdata.iterrows():

    if not math.isnan(row['Longitude']) and not math.isnan(row['Latitude']):

        mc.add_child(Marker([row['Latitude'], row['Longitude']]))

m_2.add_child(mc)



embed_map(m_2, 'm_2.html')