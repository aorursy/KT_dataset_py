import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('../input/data-police-shootings/fatal-police-shootings-data.csv')

data.head() # Read to data set.
data.info()
# Let's check for any missing data.

data.isnull().any()
import missingno as msno

msno.matrix(data);
data.dropna(inplace=True)

data.isnull().any()

# Yes, we have solved this too.
import plotly.express as px

fig = px.pie(data['race'],names=data['race'],color_discrete_sequence=px.colors.sequential.Magenta)

fig.show()
values = data['state'].value_counts().keys().tolist()

import plotly.express as px

fig = px.choropleth(locations=values, locationmode="USA-states", color=data['state'].value_counts(), scope="usa")

fig.show()
fig = px.histogram(data,x='manner_of_death',color='gender')

fig.show()
fig = px.pie(data['body_camera'],names=data['body_camera'],color_discrete_sequence=px.colors.sequential.Rainbow_r,hole=0.5)

fig.show()
N_C = data[data['body_camera'] == False]

colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen']

fig = px.pie(N_C['race'],names=N_C['race'],hole=0.3)

fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,

                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))

fig.show()
fig = px.bar(data['flee'].value_counts())

fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)

fig.update_layout(title_text='Escape status')

fig.show()
from warnings import filterwarnings

filterwarnings('ignore')

histo = data[['date']]

histo['Kill'] = 1

histo_x=histo.groupby('date').sum()

histo_x = histo_x.reset_index()



fig = px.line(histo_x, x='date', y='Kill',color_discrete_sequence=px.colors.sequential.Rainbow)

fig.show()