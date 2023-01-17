# - Libraries -

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



import plotly.graph_objects as go

import plotly.express as px
# - Data -

url = '../input/earthquakenepal/ISCOpenquake.csv'

data = pd.read_csv(url, header='infer')
#Shape

print("Total Records: ", data.shape[0])
#Drop EventId Column

data = data.drop('eventID',axis=1)
#Inspect

data.head()
#Seperate Dataframe to get aggregate earthquake count per year

agg_func = {'latitude':'first', 'longitude':'first', 'magnitude':'count'}

df_year_count = data.groupby(['year']).agg(agg_func)



#Convert Index to Column

df_year_count['year'] = df_year_count.index



#Drop Previous Index

df_year_count.reset_index(drop=True, inplace=True)
fig = go.Figure(data=go.Scatter(x=df_year_count['year'], y=df_year_count['magnitude'], name='Earthquakes'))

fig.update_layout(title="Number of Earthquakes in Nepal over the years")

fig.show()
#Seperate Dataframe to get aggregate earthquake per year

agg_func = {'latitude':'first', 'longitude':'first', 'magnitude':'first'}

df_year_eq = data.groupby(['year']).agg(agg_func)



#Convert Index to Column

df_year_eq['year'] = df_year_eq.index



#Drop Previous Index

df_year_eq.reset_index(drop=True, inplace=True)
fig = px.density_mapbox(df_year_eq, lat='latitude', lon='longitude', z='magnitude', radius=10,

                        zoom=5,mapbox_style="stamen-terrain")

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
#Earthquakes in Nepal over the years (animated)

fig = px.scatter_geo(df_year_eq, lat='latitude', lon='longitude', color = 'magnitude',

                     hover_name='magnitude', size="magnitude",

                     animation_frame="year", center=dict(lat=30.7593, lon=80.2794)

                    )

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()