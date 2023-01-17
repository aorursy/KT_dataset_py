# linear algebra
import numpy as np 

# data processing
import pandas as pd 

# time
import datetime

# import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

# map
import folium
df = pd.read_csv('../input/nypd-motor-vehicle-collisions.csv')
df.head()
# last week 
df['DATE'] = pd.to_datetime(df['DATE'], format = '%Y-%m-%d')
n = 7
week = df['DATE'].max() - datetime.timedelta(days = n)
last_week = df[df['DATE'] > week]
incidents_per_day = last_week['DATE'].groupby(last_week['DATE']).count()
data = [go.Scatter(x=incidents_per_day.index, y=incidents_per_day.values)]

layout = dict(title = 'Number of incidents per day',
              xaxis= dict(ticklen= 2))

fig = dict(data = data, layout = layout)
iplot(fig)
incidents_by_borough = last_week['BOROUGH'].groupby(last_week['BOROUGH']).count()
data = [go.Bar(x=incidents_by_borough.index, y=incidents_by_borough.values)]

layout = dict(title = 'Number of incidents by borough',
              xaxis= dict(ticklen= 2))

fig = dict(data = data, layout = layout)
iplot(fig)
injured_killed = last_week[['DATE', 'NUMBER OF PERSONS INJURED', 'NUMBER OF PERSONS KILLED']].groupby(last_week['DATE']).sum()
killed = go.Scatter(x=injured_killed.index, y=injured_killed['NUMBER OF PERSONS KILLED'].values, name = 'killed')
injured = go.Scatter(x=injured_killed.index, y=injured_killed['NUMBER OF PERSONS INJURED'].values, name = 'injured')
layout = dict(title = 'Number of injured and killed per day',
              xaxis= dict(ticklen= 2))

fig = dict(data = [killed, injured], layout = layout)
iplot(fig)
collision_cause = last_week['CONTRIBUTING FACTOR VEHICLE 1'].append(last_week['CONTRIBUTING FACTOR VEHICLE 2']).T
collision_cause = collision_cause.groupby(collision_cause).count().sort_values(ascending=False)
collision_cause = collision_cause.drop('Unspecified')
data = [go.Bar(x=collision_cause.values, y=collision_cause.index, orientation = 'h')]

layout = dict(title = 'Number of incidents by borough',
              xaxis= dict(ticklen= 2))

fig = dict(data = data, layout = layout)
iplot(fig)
last_week_map = last_week.dropna(subset=['LATITUDE', 'LONGITUDE'])
map = folium.Map(location=[40.767937,-73.982155 ], zoom_start=10) 

for collision in last_week_map[0:100].iterrows():
    folium.Marker([collision[1]['LATITUDE'],
                   collision[1]['LONGITUDE']]).add_to(map)
    
display(map)