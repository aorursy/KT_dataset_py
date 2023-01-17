import plotly.plotly as py

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot



import numpy as np

import pandas as pd 
df = pd.read_csv('../input/consumer_complaints.csv', low_memory=False)
new_df = df.groupby(["state"]).size().reset_index(name="Number_Complaints")

new_df.head()
init_notebook_mode(connected=True)

for col in new_df.columns:

    new_df[col] = new_df[col].astype(str)



    scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\

            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]



new_df['text'] = new_df['state'] + '<br>' + 'Complaints '+new_df['Number_Complaints']

    



data = [ dict(

        type='choropleth',

        colorscale = scl,

        autocolorscale = False,

        locations = new_df['state'], 

        locationmode = 'USA-states',

        z = new_df['Number_Complaints'].astype(float),

        text = new_df['text'],

        marker = dict(

            line = dict (

                color = 'rgb(255,255,255)',

                width = 2

            ) ),

        colorbar = dict(

            title = "Number of Complaints")

        ) ]



layout = dict(

        title = 'Number of Complaints by State<br>',

        geo = dict(

            scope='usa',

            projection=dict( type='albers usa' ),

            showlakes = False,

            lakecolor = 'rgb(255, 255, 255)'),

             )

    

fig = dict( data=data, layout=layout )

iplot( fig, filename='d3-cloropleth-map' )