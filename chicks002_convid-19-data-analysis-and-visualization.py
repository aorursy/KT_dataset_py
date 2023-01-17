# load libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import folium
import geopandas as gpd
import folium 
from folium import plugins
plt.style.use("fivethirtyeight")# for pretty graphs

import plotly as py
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df_confirmed = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')
df_all = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
df_recovered = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')
df_full = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
df_full.tail(10)
cntagg = pd.DataFrame(df_all.groupby(['Country/Region'])['Confirmed'].mean())
cntagg.sort_values('Confirmed', ascending=False)
cntagg.head()
df_all['datetime'] = df_all['ObservationDate']
df_all['datetime'] = df_all['datetime'].apply(lambda x:datetime.strptime(str(x),'%m/%d/%Y'))
# lets add new fearures
df_all['month']=df_all['datetime'].apply(lambda x:x.month)
df_all['day']=df_all['datetime'].apply(lambda x:x.day)
df_all['year']=df_all['datetime'].apply(lambda x:x.year)
df_all['week']=df_all['datetime'].apply(lambda x:x.week)
df_all['state'] = df_all['Province/State']
df_all['country']= df_all['Country/Region']
df_all.drop(columns=['ObservationDate','Province/State','Country/Region'], inplace=True)
df_all.sample(5)
### count,season,hour
plt.figure(figsize=(15,8))
weekagg = pd.DataFrame(df_all.groupby(['week','month'])['Deaths'].mean()).reset_index()
sns.pointplot(data=weekagg,x=weekagg['week'],y=weekagg['Deaths'],hue=weekagg['week']).set(title='Week Vs Death Counts')
plt.figure(figsize=(15,10))
cntagg = pd.DataFrame(df_all.groupby(['country','state'])['Confirmed'].mean()).reset_index()
ax = sns.barplot(x="country", y="Confirmed", data=cntagg.sort_values('Confirmed', ascending=False)[:10])
plt.title('Top 10 Countries with Covid-19 Cases')
### country vs deaths
plt.figure(figsize=(18,5))
cntagg = pd.DataFrame(df_all.groupby(['country','state'])['Deaths'].mean()).reset_index()
sns.pointplot(data=cntagg,x='country',y='Deaths').set(title='Country Vs Death Counts')
### country vs recovered cases
plt.figure(figsize=(18,5))
cntagg = pd.DataFrame(df_all.groupby(['country','state'])['Confirmed'].mean()).reset_index()
sns.scatterplot(data=cntagg,x='country',y='Confirmed').set(title='Country Vs Confirmed cases')
map_osm = folium.Map(location=[30.9756,112.2707],zoom_start=3.5,tiles='Stamen Toner')

for lat, lon, value, name in zip(df_confirmed['Lat'], df_confirmed['Long'], df_all['Confirmed'], df_confirmed['Country/Region']):
    folium.CircleMarker([lat, lon],
                        radius=value*0.1,
                        popup = ('<strong>Country</strong>: ' + str(name).capitalize() + '<br>'
                                '<strong>Confirmed Cases</strong>: ' + str(value) + '<br>'),
                        color='blue',
                        
                        fill_color='blue',
                        fill_opacity=0.3 ).add_to(map_osm)
    

map_osm

x = df_all.groupby('country')['Confirmed'].sum().sort_values(ascending=False).to_frame()
x.style.background_gradient(cmap='Reds')
x.head()
# Manipulate Dataframe
df_countries = df_all.groupby(['country', 'datetime']).sum().reset_index().sort_values('datetime', ascending=False)
df_countries = df_countries.drop_duplicates(subset = ['country'])
df_countries = df_countries[df_countries['Confirmed']>0]
df_countries.head(10)
update_date = str(df_countries.iloc[0].datetime).split(" ")[0]
# Create the Choropleth
fig = go.Figure(data=go.Choropleth(
    locations = df_countries['country'],
    locationmode = 'country names',
    z = df_countries['Confirmed'],
    colorscale = 'Reds',
    marker_line_color = 'black',
    marker_line_width = 0.5,
))
fig.update_layout(
    title_text = 'Confirmed Cases as of ' + update_date,
    title_x = 0.5,
    geo=dict(
        showframe = False,
        showcoastlines = False,
        projection_type = 'equirectangular'
    )
)
# Manipulating the original dataframe
df_countrydate = df_all[df_all['Confirmed']>0]
df_countrydate = df_countrydate.groupby(['datetime','country']).sum().reset_index()
df_countrydate['Date'] = df_countrydate['datetime'].apply(lambda x:x.strftime("%d-%m-%Y"))
df_countrydate.sort_values('datetime', ascending=False).head(10)
# Creating the visualization
fig = px.choropleth(df_countrydate, 
                    locations="country", 
                    locationmode = "country names",
                    color="Confirmed", 
                    hover_name="country", 
                    animation_frame='Date'
                   )
fig.update_layout(
    title_text = 'Global Spread of Coronavirus upto date -' + update_date,
    title_x = 0.5,
    geo=dict(
        showframe = False,
        showcoastlines = False,
    ))
    
fig.show()