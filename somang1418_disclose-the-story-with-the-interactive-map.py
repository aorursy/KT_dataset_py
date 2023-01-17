import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

%matplotlib inline 

import seaborn as sns 

import warnings

warnings.filterwarnings('ignore') 

from scipy import stats, linalg

from matplotlib import rcParams

import scipy.stats as st

import folium 

from folium import plugins

from folium.plugins import HeatMap

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.figure_factory as ff

import plotly.tools as tls

import datetime

import re



pd.set_option('display.max_columns', None)

sns.set_style('whitegrid')

warnings.filterwarnings('ignore') 
raw_data = pd.read_csv('../input/MissingMigrants-Global-2019-03-29T18-36-07.csv')
raw_data.head(5)

print(raw_data[raw_data['Location Coordinates'].isin([np.nan, np.inf, -np.inf])]) #checking None, Inf, and -Inf values

df=raw_data.dropna(subset=['Location Coordinates']) #drop it
# formatting (split by comma and map it as a column of df)



df['lat']=df['Location Coordinates'].astype('str').apply(lambda x: x.split(',')).map(lambda x: x[0]).astype('float64') 

df['long']=df['Location Coordinates'].astype('str').apply(lambda x: x.split(',')).map(lambda x: x[-1]).astype('float64')
incident_map = folium.Map(location = [df['lat'].mean(), df['long'].mean()], zoom_start = 2)

lat_long_data = df[['lat', 'long']].values.tolist()

cluster_map = folium.plugins.FastMarkerCluster(lat_long_data).add_to(incident_map)
incident_map 
#we can confirm by this pie chart that the majority of reported incidents in the dataset are from Mediterranean,US-Mexico Border,and North Africa regions. 



labels = list(raw_data['Region of Incident'].value_counts().index.values)

values = list(raw_data['Region of Incident'].value_counts().values)



trace = go.Pie(labels=labels, values=values)



iplot([trace], filename='basic_pie_chart')
# heatmap to see the map by total dead and missing 



base_map =  folium.Map(location = [df['lat'].mean(), df['long'].mean()], zoom_start = 3)

HeatMap(data=df[['lat', 'long', 'Total Dead and Missing']].groupby(['lat', 'long']).sum().reset_index().values.tolist(), radius=8, max_zoom=13).add_to(base_map)
base_map
raw_data['datetime']=pd.to_datetime(raw_data['Reported Month'].str.cat(raw_data['Reported Year'].astype('str'), sep=' '),format="%b %Y")



d1 = raw_data['datetime'].value_counts().sort_index()

data = [go.Scatter(x=d1.index, y=d1.values, name='total incident')]

layout = go.Layout(dict(title = "Counts of Incident by Reported Date",

                  xaxis = dict(title = 'month year'),

                  yaxis = dict(title = 'Incident Count'),

                  ),legend=dict(

                orientation="v"))

iplot(dict(data=data, layout=layout))
time=pd.crosstab(raw_data['Region of Incident'],raw_data['datetime']).columns

values=pd.crosstab(raw_data['Region of Incident'],raw_data['datetime']).values


data = [go.Scatter(x=time, y=values[1], name='Central America'), go.Scatter(x=time, y=values[5], name='Horn of Africa'),

       go.Scatter(x=time, y=values[6], name='Mediterranean'),go.Scatter(x=time, y=values[8], name='North Africa'),

     go.Scatter(x=time, y=values[13], name='Sub-Saharan Africa'),go.Scatter(x=time, y=values[14], name='US-Mexico Border')]

layout = go.Layout(dict(title = "Counts of Incident by Reported Date by Region (Top 6)",

                  xaxis = dict(title = 'month year'),

                  yaxis = dict(title = 'Incident Count'),

                  ),legend=dict(

                orientation="v"))

iplot(dict(data=data, layout=layout))
migration_flow=pd.crosstab(raw_data['Region of Incident'],raw_data['Migration Route']) 


data = dict(

    type='sankey',

    node = dict(

      pad = 15,

      thickness = 20,

      line = dict(

        color = "black",

        width = 0.5

      ),

      label = list(migration_flow.index) + list(migration_flow.columns),

      color = ["blue", "blue", "blue", "blue", "blue", "blue","blue","blue","blue"]

    ),

    link = dict(

      source = [0,0,0,0,1,1,2,2,2,2,3,4,4,4,5,6,6,7,8,9  ],

      target = [11,15,17,21,12,14,10,19,23,18,13,16,24,22,11,14,20,22,12 ],

      value = [2,2,1,1,248,1,51,9,64,15,499,230,255,14,1,6,1,1,1259  ]

  ))



layout =  dict(

    title = "Migration Route",

    font = dict(

      size = 12.5

    )

    

)



fig = dict(data=[data], layout=layout)

iplot(fig, validate=False)

reason_df=pd.DataFrame()

reason_df['reason']=list(raw_data['Cause of Death'].value_counts().index.values)[:19]

reason_df['num']=list(raw_data['Cause of Death'].value_counts().values)[:19]



plt.figure(figsize=(15,10))

sns.barplot(x="num", y="reason", label='small', data=reason_df.sort_values(by="num",ascending=False))

plt.title('Cause of Death', fontsize=20)

plt.tight_layout()

plt.tick_params(labelsize=20)

plt.show()

reason_region=pd.crosstab(raw_data['Region of Incident'],raw_data['Cause of Death']).loc[raw_data['Region of Incident'].value_counts().index.values[:6]][reason_df['reason'].values]

reason_region=reason_region.iloc[:,:6] #top 6 reason


trace1 = go.Bar(

    x=list(reason_df['reason'].values[:6]),

    y=reason_region.iloc[0].values,

    name=reason_region.iloc[0].name

)



trace2 = go.Bar(

    x=list(reason_df['reason'].values[:6]),

    y=reason_region.iloc[1].values,

    name=reason_region.iloc[1].name

)



trace3 = go.Bar(

    x=list(reason_df['reason'].values[:6]),

    y=reason_region.iloc[2].values,

    name=reason_region.iloc[2].name

)



trace4 = go.Bar(

    x=list(reason_df['reason'].values[:6]),

    y=reason_region.iloc[3].values,

    name=reason_region.iloc[3].name



)



trace5 = go.Bar(

    x=list(reason_df['reason'].values[:6]),

    y=reason_region.iloc[4].values,

    name=reason_region.iloc[4].name



)



trace5 = go.Bar(

    x=list(reason_df['reason'].values[:6]),

    y=reason_region.iloc[5].values,

    name=reason_region.iloc[5].name



)





data = [trace1, trace2,trace3, trace4, trace5]



layout = go.Layout(

    barmode='stack'

)





fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='stack-bar')