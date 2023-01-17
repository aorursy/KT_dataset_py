#importing necessary libraries

import pandas as pd

import numpy as np 

import plotly.offline as py

import plotly.graph_objects as go

import matplotlib.pyplot as plt

import seaborn as sns

from functools import reduce

from geopy.geocoders import Nominatim

import folium

from folium.plugins import HeatMap

sns.set_style('white')
data_country = pd.read_csv('../input/global-land-temperatures/GlobalLandTemperaturesByCity.csv', parse_dates=[0] )
data_country.head()
data_ind = data_country[data_country['Country']=='India']

data_ind.head()
data_ind.shape
data_ind.info()
data_ind.describe([.25,.50,.75,.90,.95])
#univariate analysis for average temperature

sns.distplot(data_ind['AverageTemperature'])
sns.distplot(data_ind['AverageTemperatureUncertainty'])
#data from 1816 to 2013 - showing the range of temperatures for few citites

def boxplt(i):

  sns.boxplot(x=data_ind[data_ind['City']==i]['City'], y=data_ind[data_ind['City']==i]['AverageTemperature'], data=data_ind, color='c')

  plt.tight_layout()
plt.figure(figsize=(30,15))



for l,i in enumerate(['Bombay','New Delhi','Calcutta','Ahmadabad','Bangalore', 'Chandigarh','Gandhinagar','Jammu','Madras','Patna' ,'Ranchi','Thiruvananthapuram','Port Blair']):

  plt.subplot(2,7,l+1)

  boxplt(i)
century_temp_18 = data_ind[(data_ind['dt']>='1800-01-01') & (data_ind['dt']<'1900-01-01')]

century_temp_18 = century_temp_18.groupby('City').agg('mean')



century_temp_19 = data_ind[(data_ind['dt']>='1900-01-01') & (data_ind['dt']<'2000-01-01')]

century_temp_19 = century_temp_19.groupby('City').agg('mean')



century_temp_20 = data_ind[(data_ind['dt']>='2000-01-01') & (data_ind['dt']<'2100-01-01')]

century_temp_20 = century_temp_20.groupby('City').agg('mean')
l1 = [century_temp_18, century_temp_19, century_temp_20]



century_comb = reduce(lambda left,right: pd.merge(left,right,on=['City'], suffixes=('_18','_19')), l1)

century_comb.columns = ['AverageTemperature_18','AverageTemperatureUncertainty_18', 'AverageTemperature_19','AverageTemperatureUncertainty_19',

                        'AverageTemperature_20','AverageTemperatureUncertainty_20']
century_comb['Diff18_20'] = [y-i for i,y in zip(century_comb['AverageTemperature_18'], century_comb['AverageTemperature_20'])]

order_century_comb = century_comb.sort_values(by='Diff18_20', ascending=False)

print(order_century_comb[:5].index)

print(order_century_comb[-6:-1].index)
fig = go.Figure()

fig.add_trace(go.Scatter(y=century_temp_18['AverageTemperature'], x=century_temp_18.index,name='18th Century'))

fig.add_trace(go.Scatter(y=century_temp_19['AverageTemperature'], x=century_temp_19.index, name='19th Century'))

fig.add_trace(go.Scatter(y=century_temp_20['AverageTemperature'], x=century_temp_20.index, name='20th Century'))

fig.update_xaxes(rangeslider_visible=True, title_text = 'Cities')

fig.update_yaxes(title_text='Mean Temperature')

fig.update_layout(title = 'Mean Temperature For Cities (Every Century)', height=800, width=1500)

fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(y=century_comb['Diff18_20'], x=century_comb.index,name='Temperature'))



fig.update_xaxes(rangeslider_visible=True, title_text = 'Cities')

fig.update_yaxes(title_text='Mean Temperature')

fig.update_layout(title = 'Mean Temperature Increase from 18th to 20th Century (For Cities)', height=800, width=1500)

fig.show()
lat_lon = []

geolocater = Nominatim(user_agent='app',timeout=4)

for location in century_comb.index:

  location = geolocater.geocode(location)

  if location is None:

    lat_lon.append(np.nan)

  else:

    geo = (location.latitude, location.longitude)

    lat_lon.append(geo)



century_comb['geo'] = lat_lon
century_comb_copy = century_comb.copy()

century_comb_copy.dropna(inplace=True)

lat, lon = zip(*np.array(century_comb_copy['geo']))

century_comb_copy['lat'] = lat

century_comb_copy['lon'] = lon 

basemap = folium.Map(control_scale=True, zoom_start=5, location=(20.5937, 78.9629))

HeatMap(century_comb_copy[['lat','lon', 'Diff18_20']].values.tolist(), radius=15).add_to(basemap)



basemap
decade_19_1 = data_ind[(data_ind['dt']>='1910-01-01') & (data_ind['dt']<'1920-01-01')]

decade_19_1 = decade_19_1.groupby('City').agg('mean')





decade_19_2 = data_ind[(data_ind['dt']>='1920-01-01') & (data_ind['dt']<'1930-01-01')]

decade_19_2 = decade_19_2.groupby('City').agg('mean')



decade_19_3 = data_ind[(data_ind['dt']>='1930-01-01') & (data_ind['dt']<'1940-01-01')]

decade_19_3 = decade_19_3.groupby('City').agg('mean')



decade_19_4 = data_ind[(data_ind['dt']>='1940-01-01') & (data_ind['dt']<'1950-01-01')]

decade_19_4 = decade_19_4.groupby('City').agg('mean')



decade_19_5 = data_ind[(data_ind['dt']>='1950-01-01') & (data_ind['dt']<'1960-01-01')]

decade_19_5 = decade_19_5.groupby('City').agg('mean')



decade_19_6 = data_ind[(data_ind['dt']>='1960-01-01') & (data_ind['dt']<'1970-01-01')]

decade_19_6 = decade_19_6.groupby('City').agg('mean')



decade_19_7 = data_ind[(data_ind['dt']>='1970-01-01') & (data_ind['dt']<'1980-01-01')]

decade_19_7 = decade_19_7.groupby('City').agg('mean')



decade_19_8 = data_ind[(data_ind['dt']>='1980-01-01') & (data_ind['dt']<'1990-01-01')]

decade_19_8 = decade_19_8.groupby('City').agg('mean')



decade_19_9 = data_ind[(data_ind['dt']>='1990-01-01') & (data_ind['dt']<'2000-01-01')]

decade_19_9 = decade_19_9.groupby('City').agg('mean')



decade_20_1 = data_ind[(data_ind['dt']>='2000-01-01') & (data_ind['dt']<'2010-01-01')]

decade_20_1 = decade_20_1.groupby('City').agg('mean')



decade_20_2 = data_ind[(data_ind['dt']>='2010-01-01') & (data_ind['dt']<'2020-01-01')]

decade_20_2 = decade_20_2.groupby('City').agg('mean')
fig = go.Figure()

fig.add_trace(go.Scatter(y=decade_19_1['AverageTemperature'], x=decade_19_1.index,name='1910'))

fig.add_trace(go.Scatter(y=decade_19_2['AverageTemperature'], x=decade_19_2.index, name='1920'))

fig.add_trace(go.Scatter(y=decade_19_3['AverageTemperature'], x=decade_19_3.index, name='1930'))

fig.add_trace(go.Scatter(y=decade_19_4['AverageTemperature'], x=decade_19_4.index, name='1940'))

fig.add_trace(go.Scatter(y=decade_19_5['AverageTemperature'], x=decade_19_5.index, name='1950'))

fig.add_trace(go.Scatter(y=decade_19_6['AverageTemperature'], x=decade_19_6.index, name='1960'))

fig.add_trace(go.Scatter(y=decade_19_7['AverageTemperature'], x=decade_19_7.index, name='1970'))

fig.add_trace(go.Scatter(y=decade_19_8['AverageTemperature'], x=decade_19_8.index, name='1980'))

fig.add_trace(go.Scatter(y=decade_19_9['AverageTemperature'], x=decade_19_9.index, name='1990'))

fig.add_trace(go.Scatter(y=decade_20_1['AverageTemperature'], x=decade_20_1.index, name='2000'))

fig.add_trace(go.Scatter(y=decade_20_2['AverageTemperature'], x=decade_20_2.index, name='2010'))



fig.update_xaxes(rangeslider_visible=True, title_text = 'Cities')

fig.update_yaxes(title_text='Mean Temperature')

fig.update_layout(title = 'Mean Temperature For Cities (Every Decade)', height=800, width=1500)

fig.show()
l1 = [decade_19_1,decade_19_2,decade_19_3,decade_19_4,decade_19_5,decade_19_6,decade_19_7,decade_19_8,decade_19_9,decade_20_1,decade_20_2]



decade_comb = reduce(lambda left,right: pd.merge(left,right,on=['City']), l1)

decade_comb.columns = ['AverageTemperature_1','AverageTemperatureUncertainty_1', 'AverageTemperature_2','AverageTemperatureUncertainty_2',

                        'AverageTemperature_3','AverageTemperatureUncertainty_3', 'AverageTemperature_4','AverageTemperatureUncertainty_4', 

                        'AverageTemperature_5','AverageTemperatureUncertainty_5', 'AverageTemperature_6','AverageTemperatureUncertainty_6',

                        'AverageTemperature_7','AverageTemperatureUncertainty_7', 'AverageTemperature_8','AverageTemperatureUncertainty_8',

                        'AverageTemperature_9','AverageTemperatureUncertainty_9', 'AverageTemperature_10','AverageTemperatureUncertainty_10',

                        'AverageTemperature_11','AverageTemperatureUncertainty_11']
decade_comb['Diff1_11'] = [y-i for i,y in zip(decade_comb['AverageTemperature_1'], decade_comb['AverageTemperature_11'])]

order_decade_comb = decade_comb.sort_values(by='Diff1_11', ascending=False)

fig = go.Figure()

fig.add_trace(go.Scatter(y=decade_comb['Diff1_11'], x=decade_comb.index,name='Temperature'))



fig.update_xaxes(rangeslider_visible=True, title_text = 'Cities')

fig.update_yaxes(title_text='Mean Temperature')

fig.update_layout(title = 'Mean Temperature Increase every Decade (For Cities)', height=800, width=1500)

fig.show()
print(order_decade_comb[:5].index)

print(order_decade_comb[-6:-1].index)
decade_comb['lat']  = century_comb_copy['lat'] 

decade_comb['lon']= century_comb_copy['lon'] 

decade_comb_copy = decade_comb.copy()

decade_comb_copy.dropna(inplace=True)



basemap = folium.Map(control_scale=True, zoom_start=5, location=(20.5937, 78.9629))

HeatMap(decade_comb_copy[['lat','lon', 'Diff1_11']].values.tolist(), radius=15).add_to(basemap)



basemap
date_range = pd.date_range('2003-01-01', '2013-01-01', freq='YS')

last_decade = pd.DataFrame(index=century_comb.index)



for y,i in enumerate(date_range):

  last_decade_y = data_ind[(data_ind['dt']>=i) & (data_ind['dt']<i.replace(year= i.year + (y+1)))]

  last_decade_y = last_decade_y.groupby('City').agg('mean')

  last_decade = last_decade.merge(last_decade_y,on=['City'])
last_decade.columns = ['AverageTemperature2003', 'AverageTemperatureUncertainity2003','AverageTemperature2004', 'AverageTemperatureUncertainity2004',

                       'AverageTemperature2005', 'AverageTemperatureUncertainity2005','AverageTemperature2006', 'AverageTemperatureUncertainity2006',

                       'AverageTemperature2007', 'AverageTemperatureUncertainity2007','AverageTemperature2008', 'AverageTemperatureUncertainity2008',

                       'AverageTemperature2009', 'AverageTemperatureUncertainity2009','AverageTemperature2010', 'AverageTemperatureUncertainity2010',

                       'AverageTemperature2011', 'AverageTemperatureUncertainity2011','AverageTemperature2012', 'AverageTemperatureUncertainity2012',

                       'AverageTemperature2013', 'AverageTemperatureUncertainity2013',]
last_decade['diff03_13'] =  last_decade['AverageTemperature2013'] - last_decade['AverageTemperature2003'] 

order_last_decade = last_decade.sort_values(by='diff03_13', ascending=False)



fig = go.Figure()

fig.add_trace(go.Scatter(y=last_decade['AverageTemperature2003'], x=last_decade.index,name='2003'))

fig.add_trace(go.Scatter(y=last_decade['AverageTemperature2004'], x=last_decade.index, name='2004'))

fig.add_trace(go.Scatter(y=last_decade['AverageTemperature2005'], x=last_decade.index, name='2005'))

fig.add_trace(go.Scatter(y=last_decade['AverageTemperature2006'], x=last_decade.index, name='2006'))

fig.add_trace(go.Scatter(y=last_decade['AverageTemperature2007'], x=last_decade.index, name='2007'))

fig.add_trace(go.Scatter(y=last_decade['AverageTemperature2008'], x=last_decade.index, name='2008'))

fig.add_trace(go.Scatter(y=last_decade['AverageTemperature2009'], x=last_decade.index, name='2009'))

fig.add_trace(go.Scatter(y=last_decade['AverageTemperature2010'], x=last_decade.index, name='2010'))

fig.add_trace(go.Scatter(y=last_decade['AverageTemperature2011'], x=last_decade.index, name='2011'))

fig.add_trace(go.Scatter(y=last_decade['AverageTemperature2012'], x=last_decade.index, name='2012'))

fig.add_trace(go.Scatter(y=last_decade['AverageTemperature2013'], x=last_decade.index, name='2013'))



fig.update_xaxes(rangeslider_visible=True, title_text = 'Cities')

fig.update_yaxes(title_text='Mean Temperature')

fig.update_layout(title = 'Mean Temperature For Cities (Last Decade)', height=800, width=1500)

fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(y=last_decade['diff03_13'], x=last_decade.index,name='Temperature'))



fig.update_xaxes(rangeslider_visible=True, title_text = 'Cities')

fig.update_yaxes(title_text='Mean Temperature')

fig.update_layout(title = 'Mean Temperature Increase from 2003-2013 (For Cities)', height=800, width=1500)

fig.show()
print(order_last_decade[:5].index)

print(order_last_decade[-6:-1].index)
last_decade['lat']  = century_comb_copy['lat'] 

last_decade['lon']= century_comb_copy['lon'] 

last_decade_copy = last_decade.copy()

last_decade_copy.dropna(inplace=True)



basemap = folium.Map(control_scale=True, zoom_start=5, location=(20.5937, 78.9629))

HeatMap(last_decade_copy[['lat','lon', 'diff03_13']].values.tolist(), radius=15).add_to(basemap)



basemap