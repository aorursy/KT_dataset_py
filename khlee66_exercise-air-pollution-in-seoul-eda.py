import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline
import seaborn as sns
df_summary = pd.read_csv('/kaggle/input/air-pollution-in-seoul/AirPollutionSeoul/Measurement_summary.csv')  # A condensed dataset based on the below three data.
df_info = pd.read_csv('/kaggle/input/air-pollution-in-seoul/AirPollutionSeoul/Original Data/Measurement_info.csv')  # Air pollution measurement information
df_item = pd.read_csv('/kaggle/input/air-pollution-in-seoul/AirPollutionSeoul/Original Data/Measurement_item_info.csv')  #  Information on air pollution measurement items
df_station = pd.read_csv('/kaggle/input/air-pollution-in-seoul/AirPollutionSeoul/Original Data/Measurement_station_info.csv')  # Information on air pollution instrument stations
df_summary.head()
date_time = df_summary['Measurement date'].str.split(' ', n=1, expand=True)
date_time.head()
df_summary['date'] = date_time[0]
df_summary['time'] = date_time[1]
df_summary = df_summary.drop(['Measurement date'], axis=1)
df_summary.head()
# print(df_summary['Station code'].unique())
print("`Station code` column has {} distinct values".format(df_summary['Station code'].nunique()))

# print(df_summary['Address'].unique())
print("`Address` column has {} distinct values".format(df_summary['Address'].nunique()))

# print(df_summary['Latitude'].unique())
print("`Latitude` column has {} distinct values".format(df_summary['Latitude'].nunique()))

# print(df_summary['Longitude'].unique())
print("`Longitude` column has {} distinct values".format(df_summary['Longitude'].nunique()))
df_summary.describe()
print('We have {} negative values for SO2'.format(df_summary['SO2'].loc[df_summary['SO2'] < 0].count()))
print('We have {} negative values for NO2'.format(df_summary['NO2'].loc[df_summary['NO2'] < 0].count()))
print('We have {} negative values for O3'.format(df_summary['O3'].loc[df_summary['O3'] < 0].count()))
print('We have {} negative values for CO'.format(df_summary['CO'].loc[df_summary['CO'] < 0].count()))
print('We have {} negative values for PM10'.format(df_summary['PM10'].loc[df_summary['PM10'] < 0].count()))
print('We have {} negative values for PM2.5'.format(df_summary['PM2.5'].loc[df_summary['PM2.5'] < 0].count()))
import plotly
import plotly.graph_objs as go
import plotly.offline as py
# import plotly.io as pio
# pio.renderers.default = 'colab'
plotly.offline.init_notebook_mode(connected=True)
# 
to_drop = df_summary.loc[(df_summary['SO2']<0) | (df_summary['NO2']<0) | (df_summary['CO']<0) | (df_summary['O3']<0)]
print("Total number of to_drop records is {}".format(to_drop.shape[0]))
to_drop.head()
# drop records which contains columns having negative value
df_summary.drop(to_drop.index, axis=0, inplace=True)
# Gas pollutants
# Sampling a single station data
sample_101 = df_summary.loc[df_summary['Station code'] == 101]
data = [go.Scatter(x=sample_101['date'],
                   y=sample_101['SO2']),
        go.Scatter(x=sample_101['date'],
                   y=sample_101['NO2']),
        go.Scatter(x=sample_101['date'],
                   y=sample_101['O3']),
        go.Scatter(x=sample_101['date'],
                   y=sample_101['CO'])]

layout = go.Layout(title='Gases Levels after dropping negative value (Sample - Station code: 101)',
                   yaxis={'title': 'Level (ppm)'},
                   xaxis={'title': 'Date'})                   

fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
# check negative value drop result
gas_pollutants = df_summary[['SO2', 'NO2', 'O3', 'CO']]
gas_pollutants.apply(lambda x: x < 0).sum().sum()
# PM pollutatns
data = [go.Scatter(x=sample_101['date'],
                   y=sample_101['PM2.5']),
        go.Scatter(x=sample_101['date'],
                   y=sample_101['PM10'])]

layout = go.Layout(title='PM level (Sample - Station code: 101)',
                   yaxis={'title': 'Level (ppm)'},
                   xaxis={'title': 'Date'})                   

fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
to_drop_PM = df_summary.loc[(df_summary['PM2.5']<0) | (df_summary['PM10']<0) | 
                            (df_summary['PM2.5']==0) | (df_summary['PM10']==0)]
to_drop_PM.head()
df_summary.drop(to_drop_PM.index, axis=0, inplace=True)
# check negative and zero value drop result
pm_pollutants = df_summary[['PM2.5', 'PM10']]
pm_pollutants.apply(lambda x: x <= 0).sum().sum()
df_seoul_daily = df_summary.groupby(['date'], as_index=False).agg({
    'SO2': 'mean', 'NO2': 'mean', 'O3': 'mean', 'CO': 'mean', 'PM10': 'mean', 'PM2.5': 'mean',
})
df_seoul_daily.head()
df_seoul_daily.plot(x='date', figsize=(15,8))
df_seoul_daily[['date', 'SO2', 'NO2', 'O3', 'CO']].plot(x='date', figsize=(15,8))
df_seoul_daily[['date', 'SO2', 'NO2', 'O3']].plot(x='date', figsize=(15,8))
pollutant_corr = df_seoul_daily.corr()  # Pearson correlation
f, ax = plt.subplots(figsize=(15, 10))
cmap = sns.diverging_palette(240, 10, n=9, as_cmap=True)
sns.heatmap(pollutant_corr, cmap=cmap, annot=True, vmax=1, center=0,
            square=True, linewidth=.5)
df_item.head()
df_seoul_daily.loc[df_seoul_daily['PM10'] <= 30, 'PM10_class'] = 0
df_seoul_daily.loc[(df_seoul_daily['PM10'] > 30) & (df_seoul_daily['PM10'] <= 80), 'PM10_class'] = 1
df_seoul_daily.loc[(df_seoul_daily['PM10'] > 80) & (df_seoul_daily['PM10'] <= 150), 'PM10_class'] = 2
df_seoul_daily.loc[(df_seoul_daily['PM10'] > 151), 'PM10_class'] = 3
df_seoul_daily['PM10_class'] = df_seoul_daily['PM10_class'].astype(int)
df_seoul_daily.head()
plt.figure(figsize=(10, 8))
sns.countplot(data=df_seoul_daily, x='PM10_class', order=df_seoul_daily['PM10_class'].value_counts().index,
              palette='Set2')
data = [go.Scatter(x=sample_101['date'],
                   y=sample_101['PM10'])]

layout = go.Layout(title='PM10 Levels (Sample - Station code: 101)',
                   xaxis={'title': 'Date'},
                   yaxis={'title': 'Level (ppm)'})

fig = go.Figure(data=data, layout=layout)

# `Good` horizontal line
fig.add_shape(
    type='line',
    x0='2017-01-01',
    y0=30,
    x1='2019-12-31',
    y1=30,
    line=dict(
        color='Green',
        width=5,
        dash='dashdot'
    )
)

# `Not Bad` horizontal line
fig.add_shape(
    type='line',
    x0='2017-01-01',
    y0=80,
    x1='2019-12-31',
    y1=80,
    line=dict(
        color='Yellow',
        width=5,
        dash='dashdot'
    )
)

py.iplot(fig)
pm10_top5 = df_summary.groupby(by='Station code').agg({'PM10': 'mean'}).sort_values(by='PM10', ascending=False).head(5).reset_index()
pm10_top5_st_codes = pm10_top5['Station code'].tolist()
print('PM Top 5 station codes: {}'.format(pm10_top5_st_codes))
pm10_top5_district = df_station.set_index('Station code').loc[pm10_top5_st_codes].reset_index()['Station name(district)']
pm10_top5.insert(loc=1, column='Station name(district)', value=pm10_top5_district)
pm10_top5
plt.figure(figsize=(10, 8))
sns.barplot(data=pm10_top5, x='Station name(district)', y='PM10', palette='Set3')
import requests
import folium
import json
import random
df_station.head()
center = [37.541, 126.986] # center of Seoul
m = folium.Map(location=center, zoom_start=11) # set map

# seoul district geo json
# https://raw.githubusercontent.com/southkorea/seoul-maps/master/kostat/2013/json/seoul_municipalities_geo.json

seoul_geo_url = 'https://raw.githubusercontent.com/southkorea/seoul-maps/master/kostat/2013/json/seoul_municipalities_geo.json'
resp = requests.get(seoul_geo_url)
seoul_geo = json.loads(resp.text)

# Add GeoJson to map
# GeoJson: info for representing polygon(district, borderline)
folium.GeoJson(
    seoul_geo,
    name='seoul_municipalities',
).add_to(m)

# Add marker


for i in df_station.index[:]:
  marker_popup_str = 'Station Name: ' + str(df_station.loc[i, 'Station name(district)'])
  folium.Marker(df_station.loc[i, ['Latitude', 'Longitude']],
                popup=marker_popup_str,
                icon=folium.Icon(color='black')).add_to(m)

m                
df_item.head()
def get_criteria(df_item, poll_item):
  critera = df_item[df_item['Item name'] == poll_item].iloc[0, 3:]
  return critera

def seoul_pollutant_map(df_day, df_item, poll_item):
  """ Visualize pollutant item status of each district by color marker """

  criteria = get_criteria(df_item, poll_item)

  df_day_c = df_day.copy()

  # set color of marker
  df_day_c['color'] = ''
  df_day_c.loc[df_day_c[poll_item] <= criteria[3], 'color'] = 'red'
  df_day_c.loc[df_day_c[poll_item] <= criteria[2], 'color'] = 'orange' # yellow
  df_day_c.loc[df_day_c[poll_item] <= criteria[1], 'color'] = 'green'
  df_day_c.loc[df_day_c[poll_item] <= criteria[0], 'color'] = 'blue'

  center = [37.541, 126.986] # center of Seoul
  m = folium.Map(location=center, zoom_start=11) # set map
  
  seoul_geo_url = 'https://raw.githubusercontent.com/southkorea/seoul-maps/master/kostat/2013/json/seoul_municipalities_geo.json'
  resp = requests.get(seoul_geo_url)
  seoul_geo = json.loads(resp.text)

  # Add GeoJson to map
  folium.GeoJson(
      seoul_geo,
      name='seoul_municipalities',
  ).add_to(m)

  for i in df_day_c.index:
    marker_popup_str = 'Station : ' + str(df_day_c.loc[i, 'Station code']) + ':' + str(df_day_c.loc[i, poll_item])
    folium.Marker(df_day_c.loc[i, ['Latitude', 'Longitude']],
                  popup=marker_popup_str,
                  icon=folium.Icon(color=df_day_c.loc[i, 'color'])).add_to(m)

  return m
random.seed(5)
ind = random.randint(1, len(df_summary))

day = df_summary.loc[ind, 'date']
print(day)
df_day = df_summary[df_summary['date'] == day]
seoul_pollutant_map(df_day, df_item, 'PM10')