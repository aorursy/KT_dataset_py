import pandas as pd

import numpy as np

import plotly.plotly as py

import plotly.graph_objs as go

from plotly import tools



# These two lines are what allow your code to show up in a notebook!

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode()



import os
# Read data

locations_df = pd.read_csv('../input/incidents_locations.csv')
# Encode type column

from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

le.fit(locations_df['type'])

locations_df['type_val'] = le.transform(locations_df['type'])
# Determine co-ordinate range

lat_max = locations_df.latitude.max()

lat_min = locations_df.latitude.min()



lon_max = locations_df.longitude.max()

lon_min = locations_df.longitude.min()
# Plot map



scl = [0,"rgb(150,0,90)"],[0.125,"rgb(0, 0, 200)"],[0.25,"rgb(0, 25, 255)"],\

[0.375,"rgb(0, 152, 255)"],[0.5,"rgb(44, 255, 150)"],[0.625,"rgb(151, 255, 0)"],\

[0.75,"rgb(255, 234, 0)"],[0.875,"rgb(255, 111, 0)"],[1,"rgb(255, 0, 0)"]



data = [go.Scattergeo(

    lat = locations_df['latitude'],

    lon = locations_df['longitude'],

    text = locations_df['type'].astype(str),

    marker = dict(

        color = locations_df['type_val'],

        colorscale = 'Hot',

        #reversescale = True,

        opacity = 0.7,

        size = 5,        

        #colorbar = dict(

        #    title = 'Incident Type',

        #    titleside = 'top',

        #    tickmode = 'array',

        #    tickvals = np.arange(len(le.classes_)),

        #    ticktext = le.classes_,

        #    ticks = 'outside'

        #)                

    )

)]



layout = dict(

    geo = dict(

        scope = 'world',

        showland = True,

        showocean = False,

        showlakes = False,

        showsubunits = False,

        showcountries = True,

        showcoastlines = False,

        showframe=False,

        landcolor = "#FFFFFF",

        subunitcolor = "#FF3333",

        countrycolor = "#000000",

        oceancolor = "#0077be",

        lakecolor = "#00FFFF",

        resolution = 110,

        #projection = dict(

        #    type = "natural earth",

        #    rotation = dict(

        #        lon = -100

        #    )

        #),

        lonaxis = dict(

            #showgrid = True,

            #gridwidth = 0.5,

            range= [ lon_min, lon_max ],

            #dtick = 5

        ),

        lataxis = dict (

            #showgrid = True,

            #gridwidth = 0.5,

            range= [ lat_min, lat_max ],

            #dtick = 5

        )

    ),

    title = 'Bike-related incidents reported across the world',

    #autosize=True,

    #width=1000,

    #height=600

)



fig = dict(data=data, layout=layout)

iplot(fig)