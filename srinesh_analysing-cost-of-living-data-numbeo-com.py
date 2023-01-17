import numpy as np 
import pandas as pd 
import plotly.plotly as py
import plotly.graph_objs as go 
from plotly.offline import iplot, offline
offline.init_notebook_mode(connected=True) 
input_folder = '../input'
data2016 = pd.read_csv('/'.join([input_folder,'cost-of-living-2016.csv']))
data2017 = pd.read_csv('/'.join([input_folder,'cost-of-living-2017.csv']))
data2018 = pd.read_csv('/'.join([input_folder,'cost-of-living-2017.csv']))

data2016.head()
# Replace US state name with the country code
data2016['Country'] = data2016['Country'].apply(lambda x: 'United States' if x.isupper() else x)
data2017['Country'] = data2017['Country'].apply(lambda x: 'United States' if x.isupper() else x)
data2018['Country'] = data2018['Country'].apply(lambda x: 'United States' if x.isupper() else x)
data_country = data2016.groupby(['Country']).mean().round(1)

data = dict(type="choropleth",
           locations = data_country.index.values,
            locationmode = "country names",
            colorscale = 'Viridis',
           z = data_country['Cost.of.Living.Index'],
           text = data_country.index.values,
           colorbar = {'title':'Cost of living Index'})

layout = dict(title="Mean cost of living Index 2016",
            font=dict(family='Courier New, monospace', size=10),
             geo = dict(showframe=False,
                      projection = {'type':'equirectangular'}))
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap,validate=False)