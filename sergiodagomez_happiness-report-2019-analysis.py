import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



import plotly.plotly as py

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)



import os

print(os.listdir("../input"))
data = pd.read_csv('../input/world-happiness-report-2019.csv')

print('Rows in Data: ', data.shape[0])

print('Columns in Data: ', data.shape[1])

data.head(10)
data.tail(10)
data = data.rename({'Country (region)':'Country'}, axis=1)

data.dtypes
mask = np.zeros_like(data.corr(), dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

plt.figure(figsize = (19,15))

sns.heatmap(data.corr(), mask = mask, annot=True, cmap="YlGnBu", linewidths=.2, square=True)
map_data = [go.Choropleth(

           colorscale =  'YlGnBu',

           locations = data['Country'],

           locationmode = 'country names',

           z = data["Ladder"], 

           text = data['Country'],

           colorbar = {'title':'Ladder Rank'})]



layout = dict(title = 'Happiness distribution', 

             geo = dict(showframe = False, 

                       projection = dict(type = 'equirectangular')))



world_map = go.Figure(data=map_data, layout=layout)

iplot(world_map)
map_data = [go.Choropleth( 

           locations = data['Country'],

           locationmode = 'country names',

           z = data["Freedom"], 

           text = data['Country'],

           colorbar = {'title':'Ladder Rank'})]



layout = dict(title = 'Least Freedom', 

             geo = dict(showframe = False, 

                       projection = dict(type = 'equirectangular')))



world_map = go.Figure(data=map_data, layout=layout)

iplot(world_map)

GDP = data.sort_values(by='Log of GDP\nper capita')

GDP.head(10)
SS = data.sort_values(by='Social support')

SS.head(10)
HLE = data.sort_values(by='Healthy life\nexpectancy')

HLE.head(10)