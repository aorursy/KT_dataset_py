#Invite people for the Kaggle "After Party"

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

import seaborn as sns

import numpy as np

plt.style.use('fivethirtyeight')

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

from mpl_toolkits.basemap import Basemap

import folium

import folium.plugins

from matplotlib import animation,rc

import io

import base64

from IPython.display import HTML, display

import warnings

warnings.filterwarnings('ignore')



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# Let us gather up all of our data 

df = pd.read_csv('../input/Facility Air Pollution Dataset - All Facilities.csv')



# Let us gather our training data

df_train = pd.read_csv('../input/Train.csv')



# Let us gather our testing data

df_test = pd.read_csv('../input/Test.csv')
df.head()
map_Evansville = folium.Map(location=[37.974764, -87.555848],

                            zoom_start = 13) 



map_Evansville
from folium import plugins

from folium.plugins import HeatMap



# Ensure you're handing it floats

df['Latitude'] = df['Latitude'].astype(float)

df['Longitude'] = df['Longitude'].astype(float)



# Reducing data size so it runs faster

df = df[['Latitude', 'Longitude']]

df = df.dropna(axis=0, subset=['Latitude','Longitude'])



# List comprehension to make out list of lists

heat_data = [[row['Latitude'],row['Longitude']] for index, row in df.iterrows()]



# Plot it on the map

HeatMap(heat_data).add_to(map_Evansville)



# Display the map

map_Evansville