#Libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from kaggle_secrets import UserSecretsClient



import plotly.graph_objects as go

import plotly.express as px

#Data Load

url = '../input/data-of-india/complete.csv'

data = pd.read_csv(url, header='infer')
user_secrets = UserSecretsClient()

MapKey = user_secrets.get_secret("MapBoxKey")
#Inspect

data.head()
#Aggregate Function

agg_func = {'Latitude':'first', 'Longitude':'first', 

            'Total Confirmed cases': 'sum',

            'Death': 'sum',

            'Cured/Discharged/Migrated': 'sum' }
# New DataFrame with Aggregated Data

df = data.groupby('Name of State / UT', as_index=False).agg (agg_func)
#Renaming Columns

df = df.rename(columns={'Total Confirmed cases':'Cases', 'Cured/Discharged/Migrated': 'Recovered', 'Name of State / UT': 'State'})
#Inspect New Dataframe

df.head()
fig = px.scatter_mapbox(df, lat="Latitude", lon="Longitude", hover_name="State", hover_data=["Cases", "Death", "Recovered"],

                        color_discrete_sequence=["darkmagenta"], zoom=2.5, height=300)

fig.update_layout(mapbox_style="open-street-map")

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
px.set_mapbox_access_token(MapKey)

fig = px.scatter_mapbox(df, lat="Latitude", lon="Longitude",     color="Cases", size="Cases",

                  color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=2.5)

fig.show()
px.set_mapbox_access_token(MapKey)

fig = px.scatter_mapbox(df, lat="Latitude", lon="Longitude",     color="Death", size="Death",

                  color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=2.5)

fig.show()
px.set_mapbox_access_token(MapKey)

fig = px.scatter_mapbox(df, lat="Latitude", lon="Longitude",     color="Recovered", size="Recovered",

                  color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=2.5)

fig.show()