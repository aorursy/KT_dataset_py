#Libraries

import pandas as pd

import numpy as np



import plotly.graph_objects as go

import plotly.express as px
url = '../input/tourist-places-in-malaysia-with-coordinates/dataset tempat perlancongan Malaysia.csv'

data = pd.read_csv(url, header='infer')
#Inspect

data.head()
#Drop

data = data.drop('No', axis=1)



#Rename

data = data.rename(columns = {'Nama Tempat':'Place', 'Negeri':'State'})
#Places per States

data.groupby('State').size()
fig = px.scatter_mapbox(data, lat="Latitude", lon="Longitude", hover_name="Place", hover_data=["State"],

                        color_discrete_sequence=["darkmagenta"], zoom=5.5, height=300)

fig.update_layout(mapbox_style="open-street-map")

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()