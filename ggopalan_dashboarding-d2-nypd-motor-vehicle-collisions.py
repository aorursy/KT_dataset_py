import os
import json

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# import plotly
import plotly.plotly as py
import plotly.graph_objs as go

# these two lines are what allow your code to show up in a notebook!
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()
# Constants
COLLISIONS_CSV = '../input/nypd-motor-vehicle-collisions.csv'
METADATA_JSON = '../input/socrata_metadata.json'
DATADICT = '../input/Collision_DataDictionary.xlsx'

# Read data in
collisions = pd.read_csv(COLLISIONS_CSV)

# Slice data for plotting
latest_date = collisions.iloc[0].DATE
earliest_date = collisions.iloc[-1].DATE

# Filter the dataframes into number of injured and number of deaths
injured_df = collisions[collisions['NUMBER OF PERSONS INJURED'] > 0]
killed_df = collisions[collisions['NUMBER OF PERSONS KILLED'] > 0]
injured_df1000 = injured_df.iloc[0:1000]
killed_df1000 = killed_df.iloc[0:1000]
# specify what we want our map to look like
data = [ dict(
        type='scattergeo',
        # autocolorscale = False,
        lon = injured_df1000['LONGITUDE'],
        lat = injured_df1000['LATITUDE'],
        text = injured_df1000['NUMBER OF PERSONS INJURED'],
        locationmode = 'USA-states'
       ) ]

# chart information
layout = dict(
        title = 'Number of Injuries in Traffic Incidents',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
   
# actually show our figure
fig = dict( data=data, layout=layout )
iplot( fig, filename='d3-cloropleth-map' )
# specify what we want our map to look like
data = [ dict(
        type='scattergeo',
        # autocolorscale = False,
        lon = killed_df1000['LONGITUDE'],
        lat = killed_df1000['LATITUDE'],
        text = killed_df1000['NUMBER OF PERSONS KILLED'],
        locationmode = 'USA-states'
       ) ]

# chart information
layout = dict(
        title = 'Number of Deaths in Traffic Incidents in New York',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
   
# actually show our figure
fig = dict( data=data, layout=layout )
iplot( fig, filename='d3-cloropleth-map' )
