# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import geopandas as gpd

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

import plotly.plotly as py

from plotly import tools

import plotly.figure_factory as ff

import folium

from folium import plugins

from io import StringIO

init_notebook_mode(connected=True)

import datetime as dt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def clean_for_plot( df ):

    df['Creation Date']=pd.to_datetime(df['Creation Date'])

    current_date=max(df['Creation Date'])

    df=df[df['Creation Date']==current_date]

    df=pd.DataFrame({'lat':df['Latitude'],'lon':df['Longitude'],'day':df['Creation Date']}).dropna()

    return df;



def print_markers(df,layer,color='red'):

    for i in range(0,len(df)):

        folium.Marker([float(df.iloc[i]['lat']), float(df.iloc[i]['lon'])],icon=folium.Icon(color=color)).add_to(layer)

    return
#Garbage Carts

garbage_carts=pd.read_csv('../input/311-service-requests-garbage-carts.csv')

garbage_nan=clean_for_plot(garbage_carts)

#Lights out

lights_out=pd.read_csv('../input/311-service-requests-street-lights-one-out.csv')

lights_out_nan=clean_for_plot(lights_out)

#All lights out

all_out=pd.read_csv('../input/311-service-requests-street-lights-all-out.csv')

all_out_nan=clean_for_plot(lights_out)

#pot holes

pot_holes=pd.read_csv('../input/311-service-requests-pot-holes-reported.csv')

pot_holes.columns=pot_holes.columns.str.title()

pot_holes_nan=clean_for_plot(pot_holes)

#Sanatation code complaints

sanatation=pd.read_csv('../input/311-service-requests-sanitation-code-complaints.csv')

sanatation_nan=clean_for_plot(sanatation)

#tree trim requests

trim=pd.read_csv('../input/311-service-requests-tree-trims.csv')

trim_nan=clean_for_plot(trim)

#Abandoned Vehicles

abandoned_cars=pd.read_csv('../input/311-service-requests-abandoned-vehicles.csv')

abandoned_nan=clean_for_plot(abandoned_cars)

#Downed Trees

trees=pd.read_csv("../input/311-service-requests-tree-debris.csv")

tree_nan=clean_for_plot(trees)
print("This map shows locations for requests from ")



print('Abandoned Vehicles:'+str(max(abandoned_nan['day']))) 

print('One Light Out:'+str(max(lights_out_nan['day']))) 

print('All Lights Out:'+str(max(all_out_nan['day']))) 

print('Pot Hole Fill Request:'+str(max(pot_holes_nan['day']))) 

print('Tree Debris:'+str(max(tree_nan['day']))) 

print('Tree Trim:'+str(max(trim_nan['day']))) 

print('Sanatation Code Complaint:'+str(max(sanatation_nan['day']))) 

print('Garbage Carts:'+str(max(garbage_nan['day']))) 

#Definiting all the layers

car_map = folium.Map(location=[41.878, -87.62], height = 600, tiles='Stamen Toner', zoom_start=12)

lay_cars=folium.FeatureGroup(name="Abandoned Vehicles")

lay_light=folium.FeatureGroup(name='One Light Out')

lay_lights=folium.FeatureGroup(name='All Lights Out')

lay_holes=folium.FeatureGroup(name='Pot Hole Fill')

lay_san=folium.FeatureGroup(name='Sanatation Code')

lay_trim=folium.FeatureGroup(name="Tree Trim")

lay_trees=folium.FeatureGroup(name="Tree Debris")

lay_gar=folium.FeatureGroup(name='Garbage Carts')
#Printing the Markers

print_markers(garbage_nan,lay_gar,color='blue')

print_markers(lights_out_nan,lay_light,color='yellow')

print_markers(all_out_nan,lay_lights,color='yellow')

print_markers(pot_holes_nan,lay_holes,color='orange')

print_markers(sanatation_nan,lay_san,color='purple')

print_markers(trim_nan,lay_trim,color='green')

print_markers(tree_nan,lay_trees,color='green')

print_markers(abandoned_nan,lay_cars,color='red')

#Adding the childern to the map

car_map.add_child(lay_cars)

car_map.add_child(lay_light)

car_map.add_child(lay_lights)

car_map.add_child(lay_holes)

car_map.add_child(lay_san)

car_map.add_child(lay_trim)

car_map.add_child(lay_trees)

car_map.add_child(lay_gar)

#Printing the map

folium.LayerControl(collapsed=True).add_to(car_map)

car_map