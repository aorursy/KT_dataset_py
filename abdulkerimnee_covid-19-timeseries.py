import plotly_express as px

import plotly

import matplotlib.pyplot as plt

import pandas as pd

import geopandas as gpd

import folium 

import warnings

warnings.filterwarnings("ignore")
data=pd.read_csv('../input/covid19timeseriesdataset/time-series-19-covid-combined_csv.csv')
data['Datetime']=pd.to_datetime(data['Date'])

data.head()
missing_states = pd.isnull(data['Province/State'])

data.loc[missing_states,'Province/State'] = data.loc[missing_states,'Country/Region']
data=data.dropna()
date_max= data['Date'] == data['Date'].max()
sorted_country=data[date_max].sort_values('Confirmed',ascending=False).head(10)
px.scatter(sorted_country, x="Confirmed", y="Deaths",

           size="Confirmed", color="Country/Region", hover_name="Country/Region",

           log_x=True, size_max=55)
fig=px.bar(sorted_country,x='Confirmed',y='Country/Region',orientation='h',color='Deaths',

           color_continuous_scale=plotly.express.colors.sequential.Aggrnyl)

fig.show()
access_token = 'pk.eyJ1IjoiYWJkdWxrZXJpbW5lc2UiLCJhIjoiY2s5aThsZWlnMDExcjNkcWFmaWUxcmh3YyJ9.s-4VLvmoPQFPXdu9Mcd6pA'

px.set_mapbox_access_token(access_token)
fig = px.scatter_mapbox(

    data, lat="Lat", lon="Long",

    size="Confirmed", size_max=50,

    color="Deaths",  color_continuous_scale=px.colors.cyclical.IceFire,

    hover_name="Country/Region",           

    zoom=1,

    animation_frame="Date", animation_group="Province/State"

)

fig.layout.coloraxis.showscale = False

fig.show()
fig = px.scatter_mapbox(

    data[date_max], lat="Lat", lon="Long",

    size="Confirmed", size_max=50,

    color="Deaths", color_continuous_scale=px.colors.cyclical.IceFire,

    hover_name="Country/Region",           

    zoom=1

)

fig.layout.coloraxis.showscale = False

fig.show()
plt.figure(figsize=(16, 8))

plt.plot(data['Datetime'], data['Confirmed'], 'b.', label = 'Confirmed')

plt.plot(data['Datetime'], data['Recovered'], 'r.', label = 'Recovered')

plt.plot(data['Datetime'],data['Deaths'],'y.',label='Deaths')

plt.xlabel('DATE');plt.title('COVID-19')

plt.legend();
cols_plot = ['Confirmed', 'Deaths','Recovered','Datetime']

axes = data[cols_plot].plot(x='Datetime',marker='.', alpha=0.5, linestyle='None', figsize=(11, 9), subplots=True)

for ax in axes:

    ax.set_ylabel('Residual')

    ax.set_xlabel('Time(Year)')