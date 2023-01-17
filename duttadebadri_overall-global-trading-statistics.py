# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import pandas as pd # package for high-performance, easy-to-use data structures and data analysis
import numpy as np # fundamental package for scientific computing with Python
import matplotlib
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()
# import plotly.offline as py
# py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
import warnings
warnings.filterwarnings('ignore')
import plotly.tools as tls
import squarify
from mpl_toolkits.basemap import Basemap
from numpy import array
from matplotlib import cm
# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/commodity_trade_statistics_data.csv')
df.head()
df.isnull().sum()
df_ie2=df.groupby(['year'],as_index=True)['weight_kg','quantity'].agg('sum')
df_ie=df.groupby(['year'],as_index=False)['weight_kg','quantity'].agg('sum')
df_ie2.head(1)
df_ie.head(1)
df_ie2.plot(figsize=(12,6))
temp1 = df_ie[['year', 'weight_kg']] 
temp2 = df_ie[['year', 'quantity']] 
# temp1 = gun[['state', 'n_killed']].reset_index(drop=True).groupby('state').sum()
# temp2 = gun[['state', 'n_injured']].reset_index(drop=True).groupby('state').sum()
trace1 = go.Bar(
    x=temp1.year,
    y=temp1.weight_kg,
    name = 'Year with Import/Export in terms of Weight (Kg.)'
)
trace2 = go.Bar(
    x=temp2.year,
    y=temp2.quantity,
    name = 'Year with Import/Export in terms of no. items (Quantity)'
)

fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Import/Export in terms of Weight (Kg.)', 'Year with Import/Export in terms of no. items (Quantity)'))
                                                          

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
                          
fig['layout']['xaxis1'].update(title='Year')
fig['layout']['xaxis2'].update(title='Year')

fig['layout']['yaxis1'].update(title='Year with Import/Export in terms of Weight (Kg.)')
fig['layout']['yaxis2'].update(title='Year with Import/Export in terms of no. items (Quantity)')
                          
fig['layout'].update(height=500, width=1500, title='Import/Export in terms of Weight(kg.) & No. of Items')
iplot(fig, filename='simple-subplot')
df.shape
cnt_srs = df['flow'].value_counts()

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color=cnt_srs.values,
        colorscale = 'Picnic'
    ),
)

layout = go.Layout(
    title='Imports/Exports Ratio'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
offline.iplot(fig, filename="Ratio")

df_country=df.groupby(['country_or_area'],as_index=False)['weight_kg','quantity'].agg('sum')
df_country=df_country.sort_values(['weight_kg'],ascending=False)
fig, ax = plt.subplots()

fig.set_size_inches(13.7, 8.27)

sns.set_context("paper", font_scale=1.5)
f=sns.barplot(x=df_country["country_or_area"].head(20), y=df_country['weight_kg'].head(20), data=df_country)
f.set_xlabel("Name of Country",fontsize=15)
f.set_ylabel("Import/Export Amount",fontsize=15)
f.set_title('Top countries dominating Global Trade')
for item in f.get_xticklabels():
    item.set_rotation(90)
df_commodity=pd.concat([df['commodity'].str.split(', ', expand=True)], axis=1)
df_commodity.head()
temp_series = df_commodity[0].value_counts().head(20)
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))

trace = go.Pie(labels=labels, values=sizes)
layout = go.Layout(
    title='Top 20 Commodity Items to be Traded',
    width=900,
    height=900,
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
offline.iplot(fig, filename="Commodity")
df_animals=df.ix[df['category']=='01_live_animals']
df_animals.head(1)
df_animals['animal']=df_animals['commodity'].str.split(',').str[0]
cnt_srs = df_animals['animal'].value_counts()
trace = go.Bar(
    y=cnt_srs.index[::-1],
    x=cnt_srs.values[::-1],
    orientation = 'h',
    marker=dict(
        color=cnt_srs.values[::-1]
    ),
)

layout = dict(
    title='Animals Traded according to demand',
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
offline.iplot(fig, filename="Animals")
df_china=df.ix[df['country_or_area']=='China']
df_china.head(1)
df_chinaie2=df_china.groupby(['year'],as_index=True)['weight_kg','quantity'].agg('sum')
df_chinaie2.plot(figsize=(10,6))
cnt_srs = df_china['flow'].value_counts()

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color=cnt_srs.values
    ),
)

layout = go.Layout(
    title='Imports/Exports Ratio (China)'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
offline.iplot(fig, filename="Ratio")
df_china_export=df_china.ix[df_china['flow']=="Export"]
df_china_commodity=pd.concat([df_china_export['commodity'].str.split(', ', expand=True)], axis=1)
df_china_commodity.head(1)
temp_series = df_china_commodity[0].value_counts().head(20)
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))

trace = go.Pie(labels=labels, values=sizes)
layout = go.Layout(
    title='Top 20 Commodity Items Exported by China',
    width=900,
    height=900,
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
offline.iplot(fig, filename="Commodity")
df_china_import=df_china.ix[df_china['flow']=="Import"]
cnt_srs = df_china_import['category'].value_counts().head(20)
trace = go.Bar(
    y=cnt_srs.index[::-1],
    x=cnt_srs.values[::-1],
    orientation = 'h',
    marker=dict(
        color=cnt_srs.values[::-1])
)

layout = dict(
    title='Top Imported Items by China',
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
offline.iplot(fig, filename="Animals")
df_china_icommodity=pd.concat([df_china_import['commodity'].str.split(', ', expand=True)], axis=1)
df_india=df.ix[df['country_or_area']=='India']
df_indiaie2=df_india.groupby(['year'],as_index=True)['weight_kg','quantity'].agg('sum')
df_indiaie2.plot(figsize=(10,6))
cnt_srs = df_india['flow'].value_counts()

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color=cnt_srs.values
    ),
)

layout = go.Layout(
    title='Imports/Exports Ratio (India)'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
offline.iplot(fig, filename="Ratio")
df_india_export=df_india.ix[df_india['flow']=="Export"]
df_india_commodity=pd.concat([df_india_export['commodity'].str.split(', ', expand=True)], axis=1)
df_india_import=df_india.ix[df_india['flow']=="Import"]
df_india_commodity_import=pd.concat([df_india_import['commodity'].str.split(', ', expand=True)], axis=1)
temp_series = df_india_commodity[0].value_counts().head(20)
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))

trace = go.Pie(labels=labels, values=sizes)
layout = go.Layout(
    title='Top 20 Commodity Items Exported by India',
    width=900,
    height=900,
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
offline.iplot(fig, filename="Commodity")

temp_series = df_india_commodity_import[0].value_counts().head(20)
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))

trace = go.Pie(labels=labels, values=sizes)
layout = go.Layout(
    title='Top 20 Commodity Items Imported by India',
    width=900,
    height=900,
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
offline.iplot(fig, filename="Commodity")
