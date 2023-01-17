import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import plotly.express as px
import plotly.graph_objects as go
df = pd.read_csv('/kaggle/input/chipotle-locations/chipotle_stores.csv')
df.info()
def get_state(text):
    state = text.split(',')[1]
    state = state.split(' ')[1]
    return state
df['STATE'] = df['address'].map(get_state)
state_value = df['STATE'].value_counts().to_frame(name='number_of_store').reset_index().rename(columns={'index':'state'})
fig = px.choropleth(state_value, locations='state', color='number_of_store',
                    locationmode='USA-states', scope="usa", title='number of store by state')
fig.show()
px.bar(state_value, x='state', y='number_of_store',
       title='number of store by state')
fig = px.pie(state_value, values='number_of_store', names='state',
       title='number of store by state')
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()
fig = px.scatter_mapbox(df, lat='latitude', lon='longitude', zoom=3, hover_name='address',
                  mapbox_style="carto-positron", title='address stores on map')
fig.show()