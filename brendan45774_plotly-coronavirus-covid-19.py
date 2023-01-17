import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import iplot

import folium

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv('../input/covid19-us-marchseptember/united_states_covid19_by_state.csv')
df.head()
df = pd.read_csv('../input/covid19-us-marchseptember/united_states_covid19_by_state.csv')

fig = go.Figure(go.Scatter(x = df['Total Cases'], y = df['Total Deaths'],
                  name='Number of Cases & Deaths '))

fig.update_layout(title='Number of Cases & Deaths (September)',
                   plot_bgcolor='rgb(230, 230,230)',
                   showlegend=True)

fig.show()
df = pd.read_csv('../input/covid19-us-marchseptember/united_states_covid19_by_state.csv')
fig = px.area(df, x="Total Deaths", y="Total Cases", color="State Region",
	      line_group="State/Territory")
fig.show()
# create trace 1 that is 3d scatter
dataframe = pd.read_csv('../input/covid19-us-marchseptember/united_states_covid19_by_state.csv')
trace1 = go.Scatter3d(
    x=dataframe['Total Cases'],
    y=dataframe['State/Territory'],
    z=dataframe['State Region'],
    mode='markers',
    marker=dict(
        size=10,
        color='rgb(255,17,0)',                # set color to an array/list of desired values      
    )
)

data = [trace1]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0  
    )
    
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
# this is a simplier and more detailed, colored than the one at top
df = pd.read_csv('../input/covid19-us-marchseptember/united_states_covid19_by_state.csv')
fig = px.scatter_3d(df, x='Total Cases', y='State/Territory', z='State Region',
                    color='Total Deaths')
fig.show()
#This is the same, but different numbers. That way you can compare how much cases and death there are.
df = pd.read_csv('../input/covid19-us-marchseptember/10-2-20 covid19 cases and deaths.csv')
fig = px.scatter_3d(df, x='Total Cases', y='State/Territory', z='State Region',
                    color='Total Deaths')
fig.show()
''''url = pd.read_csv('../input/covid19-us-marchseptember/united_states_covid19_by_state.csv')
state_geo = f'{url}/usa-states.json'

bins = list(url['Total Deaths'].quantile([0, 0.5, 0.75, 0.90, 0.95, 1]))

map3 = folium.Map(location=[37, -102], zoom_start=4)

choropleth = folium.Choropleth(
    geo_data=state_geo,
    name='covid map',
    data=url,
    columns=['State/Territory', 'Total Deaths'],
    key_on='properties.name',
    fill_color= 'YlOrRd',
    fill_opacity=1.1,
    line_opacity=0.2,
    legend_name='Total Deaths',
    bins = bins,
    reset = True
).add_to(map3)

style_function = "font-size: 15px; font-weight: bold"
choropleth.geojson.add_child(
    folium.features.GeoJsonTooltip(['name'],style=style_function, labels=False)
)

map3'''
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/718417069ead87650b90472464c7565dc8c2cb1c/coffee-flavors.csv')

fig = go.Figure()

fig.add_trace(go.Sunburst(
    ids=df.ids,
    labels=df.labels,
    parents=df.parents,
    domain=dict(column=1),
    maxdepth=2,
    insidetextorientation='radial'
))

fig.update_layout(
    margin = dict(t=10, l=10, r=10, b=10)
)

fig.show()
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pandas as pd

df1 = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/718417069ead87650b90472464c7565dc8c2cb1c/sunburst-coffee-flavors-complete.csv')
df2 = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/718417069ead87650b90472464c7565dc8c2cb1c/coffee-flavors.csv')

fig = make_subplots(
    rows = 1, cols = 2,
    column_widths = [0.4, 0.4],
    specs = [[{'type': 'treemap', 'rowspan': 1}, {'type': 'treemap'}]]
)

fig.add_trace(
    go.Treemap(
        ids = df1.ids,
        labels = df1.labels,
        parents = df1.parents),
    col = 1, row = 1)

fig.add_trace(
    go.Treemap(
        ids = df2.ids,
        labels = df2.labels,
        parents = df2.parents,
        maxdepth = 3),
    col = 2, row = 1)

fig.update_layout(
    margin = {'t':0, 'l':0, 'r':0, 'b':0}
)

fig.show()
'''df = pd.read_csv('../input/covid19-us-marchseptember/united_states_covid19_by_state.csv')
df["covid-19"] = "covid-19" # in order to have a single root node
fig = px.treemap(df, path=['State Region', 'State/Territory'], values='Total Cases',
                  color='Total Deaths', hover_data=['Case Rate per 100000'],
                  color_continuous_scale='RdBu',
                  color_continuous_midpoint= int(np.average(df['Total Deaths'],weights = df['Total Cases'])))
                                                       
fig.show()'''