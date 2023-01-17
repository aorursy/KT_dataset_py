import pandas as pd

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
data.head(5)
data.tail(5)
data = data.rename({'Country (region)':'Country'}, axis=1)

data.dtypes
plt.figure(figsize = (16,5))

sns.heatmap(data.corr(), annot=True, linewidths=.2)
map_data = [go.Choropleth( 

           locations = data['Country'],

           locationmode = 'country names',

           z = data["Ladder"], 

           text = data['Country'],

           colorbar = {'title':'Ladder Rank'})]



layout = dict(title = 'Least Satisfied Countries', 

             geo = dict(showframe = False, 

                       projection = dict(type = 'equirectangular')))



world_map = go.Figure(data=map_data, layout=layout)

iplot(world_map)
GDP = data.sort_values(by='Log of GDP\nper capita')

GDP.head(5)
map_data = [go.Choropleth( 

           locations = data['Country'],

           locationmode = 'country names',

           z = data["Freedom"], 

           text = data['Country'],

           colorbar = {'title':'Ladder Rank'})]



layout = dict(title = 'Countries With Least Freedom', 

             geo = dict(showframe = False, 

                       projection = dict(type = 'equirectangular')))



world_map = go.Figure(data=map_data, layout=layout)

iplot(world_map)