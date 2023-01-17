import numpy as np

import pandas as pd 

import geopandas as gpd 

import matplotlib.pyplot as plt

import plotly.express as px

import seaborn as sns

from plotly.offline import init_notebook_mode, iplot, plot

import plotly.graph_objs as go



import folium

from folium import Choropleth, Circle, Marker

from folium import plugins
l_month = pd.read_csv('/kaggle/input/housing-in-london/housing_in_london_monthly_variables.csv')

#l_year = pd.read_csv('/kaggle/input/housing-in-london/housing_in_london_monthly_variables.csv')

lat_long = pd.read_csv('/kaggle/input/london-boroughs-latitude-and-longitude-data/london_borough_coordinates.csv')

lat_long.head()
l_month.shape
display('London Housing Data at month level')

display(l_month.head(5))
l_month.area.unique()
l_month.info()
l_month.describe()
def compute_missing_values(df):

    #total number of missing values

    total_missing = df.isnull().sum().sort_values(ascending=False)

    

    #calculating the percentage of missing values

    percentage_missing = (100 * df.isnull().sum() / len(df))

    

    #Missing values table - total, percentage

    table_missing = pd.concat([total_missing, percentage_missing], axis = 1, 

                              keys = ['Missing values', 'Percentage of Missing Values'])

    

    #Filtering the columns with missing values

    table_missing = table_missing[table_missing.iloc[:, 0] != 0]

    

    #Summary 

    print("Total number of columns:" + str(df.shape[1]) + "\nColumns with missing values:" +str(table_missing.shape[1]))

    

    return table_missing



missing_values = compute_missing_values(l_month)

missing_values.style.background_gradient(cmap='Reds')
#Replacing the missing values with the mean of the column

no_of_crimes_mean= l_month['no_of_crimes'].mean()

l_month = l_month.fillna({'no_of_crimes' : no_of_crimes_mean})



#Removing the missing values

l_month = l_month[l_month['houses_sold'].notna()]

l_month.isnull().sum()
#Converting string to datetime

l_month = l_month.set_index(pd.to_datetime(l_month['date']))
london = l_month[l_month['borough_flag'] == 1]

london.head()
london['approx_house_price'] = london['average_price'].mul(london['houses_sold'])
london.area.unique()
london_mean = london.groupby('area').mean().reset_index()

london_mean.head()
london = london.set_index('area').join(lat_long.set_index('area'))

london = london.reset_index()

london.head()
other = l_month[l_month['borough_flag'] == 0]

other.area.unique()
areas_of_london = ['south east', 'inner london','north east', 'north west', 'london', 'south west']

other = other.loc[~other.area.isin(areas_of_london)]

other.area.unique()
fig = px.line(london,x='date',y='average_price', color='area',title='Average Price in the Boroughs of London through time')

fig.update_layout(xaxis_type="date", yaxis_type="log", xaxis_title='Year', yaxis_title='Price (£)')

fig.show()
fig = px.line(london,x='date',y='houses_sold',color='area',title='Number of Houses Sold')

fig.update_layout(xaxis_type="date", yaxis_type="log", xaxis_title='Year', yaxis_title='Price (£)')

fig.show()
trace1 = go.Bar(

                x = london_mean.area,

                y = london_mean.houses_sold*1000,

                name = "Number of houses sold * 1000",

                marker = dict(color = 'rgba(8, 103, 103, 0.8)',

                             line=dict(color='rgb(0,0,0)',width=1)),

                text = london_mean.area)

trace2 = go.Bar(

                x = london_mean.area,

                y = london_mean.average_price,

                name = "Average Price (£)",

                marker = dict(color = 'rgba(103, 8, 8, 0.7)',

                             line=dict(color='rgb(0,0,0)',width=1)),

                text = london_mean.area)



data = [trace1,trace2]

layout = go.Layout(barmode = "group", title="Average Price and Number of houses sold * 1000 per Borough")

fig = go.Figure(data = data, layout = layout)

fig.update_xaxes(ticks="outside", tickwidth=2,tickangle=45, ticklen=10,title_text="Boroughs of London")

iplot(fig)
#importing the map shape file from london-borough-and-ward-boundaries-up-to-2014

l_map = gpd.read_file('/kaggle/input/london-borough-and-ward-boundaries-up-to-2014/London_Wards/Boroughs/London_Borough_Excluding_MHW.shp')



#Filtering out the data we need

l_map = l_map[['NAME', 'geometry']]



#l_map['NAME']==london['area']

l_map = l_map.rename(columns = {'NAME' : 'area'})

l_map['area'] = l_map['area'].str.lower()

l_map['area'] = l_map['area'].str.replace('&', 'and')



#joining

london_map = l_map.set_index('area').join(london.set_index('area'))

london_map.head()
#visualing

fig, axarr = plt.subplots(1, 2, figsize=(16, 12))

london_map.plot(column='no_of_crimes', cmap='Reds', linewidth=0.5,  ax=axarr[0], edgecolor='gainsboro', legend=True, legend_kwds={'label': "Number of crimes", 'orientation' : "horizontal"})

london_map.plot(column='average_price', cmap='Reds', linewidth=0.5, ax=axarr[1], edgecolor='gainsboro', legend=True, legend_kwds={'label': "Average Price", 'orientation' : "horizontal"})

plt.show()
London = [51.506949, -0.122876]

d_map = plugins.DualMap(location=London, tiles=None, zoom_start=10)

folium.TileLayer('openstreetmap').add_to(d_map)

folium.TileLayer('cartodbpositron').add_to(d_map)

map1 = folium.FeatureGroup(name='Crime Rate').add_to(d_map.m1)

map2 = folium.FeatureGroup(name='Average Price (£)').add_to(d_map.m2)

for lat, long, area, average_price in zip(london['lat'], london['long'], london['area'], london['average_price']):

    folium.CircleMarker([lat,long], radius=10, icon=folium.Icon(color='darkred'),popup=('Borough:'+str(area).capitalize()+' ''Average Price:'+str(average_price)+''), fill_color='darkred', fill_opacity=0.7).add_to(map1)



for lat, long, area, no_of_crimes in zip(london['lat'], london['long'], london['area'], london['no_of_crimes']):

    folium.CircleMarker([lat,long], radius=10, icon=folium.Icon(color='purple'),popup=('Borough:'+str(area).capitalize()+' ''Number of Crimes:'+str(no_of_crimes)+''), fill_color='purple', fill_opacity=0.7).add_to(map2)



folium.LayerControl(collapsed=False).add_to(d_map)



d_map
