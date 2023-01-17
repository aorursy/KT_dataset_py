import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import math

#taking input files
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
#libraries to plot
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly
plotly.offline.init_notebook_mode (connected = True)

#Calendar Heatmap
!pip install calmap
import calmap

#GEOSPATIAL LIBRARIES
import geopandas as gpd
import folium
from folium import Choropleth, Circle, Marker
from folium.plugins import HeatMap, MarkerCluster
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

%matplotlib inline
#load dataset into pandas dataframe
df = pd.read_csv('/kaggle/input/data-police-shootings/fatal-police-shootings-data.csv', parse_dates=["date"])
df.head()
#check the information luke count and datatype of each column
df.info()
#null values in each columns
df.isnull().sum()
#removing null values i.e. drop rows that has atleast one NaN value
df=df.dropna(axis=0)
df.isnull().sum()
#number of unique values in each column
df.nunique()
df.head()
df_tmp = df["body_camera"].value_counts()
fig = px.bar(df_tmp,title="Body Camera Available",color=df_tmp.index)
fig.show()
df_tmp = df["armed"].value_counts()[:10]
fig = px.bar(df_tmp,color=df_tmp.index,
             title="Top 10 Cases with Armed Types", color_discrete_sequence= px.colors.sequential.Plasma_r)
fig.show()
df_tmp = df[df['armed']=='unarmed']['race']
fig = px.histogram(df_tmp,x='race',title="Unarmed People Shoot vs Race",color='race',color_discrete_sequence=px.colors.qualitative.T10)
fig.show()

df["manner_of_death"].value_counts()
df_tmp=df[df['manner_of_death']=='shot']
fig = px.histogram(df_tmp,x='race',title="Ethinicity vs Races",color='race',color_discrete_sequence=px.colors.qualitative.T10)
fig.show()
df_tmp=df["age"]
fig = px.histogram(df_tmp,histnorm='probability density', title="Probability Density of Age")
fig.show()
df_date_group = df.groupby(df["date"])
incidents = df_date_group["id"].count()
print(incidents.max(), " is the maximum no. of incidents  happened in a day")
print(incidents.min(), " is the minimum no. of incident/s happened in a day")
#each day heatmap showing the number of cases
fig,ax = calmap.calendarplot(incidents, monthticks=1, daylabels='MTWTFSS',
                    fillcolor='grey', linewidth=1,
                    fig_kws=dict(figsize=(15,15)))

#fig.colorbar(ax[0].get_children()[1],ax=ax, cmap=plt.cm.get_cmap('Reds', 9), orientation='horizontal',label='Number of incidents')
#due to 2116 unique values of cities, the geocoder takes too much time and shows timeout error, 
#so i used google sheets Add-ons i.e. geocode by Awesome Table to geocode the values.
"""geolocator = Nominatim(user_agent="my_application")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
df1['location'] = df1['city'].apply(geocode)
df1['latitude'] = df1['location'].apply(lambda loc: loc.latitude if loc else None)
df1['longitude'] = df1['location'].apply(lambda loc: loc.longitude if loc else None)
df1.head()
"""
print("city geocoded file is updated in datasets i.e. cities.csv")
#dataframe containing lat long values for the centroid of all counties in US
city_df = pd.read_csv("/kaggle/input/cities-geocoded-for-data-police-shootings/cities.csv")
city_df.drop(["Unnamed: 0","geom","address"], axis=1, inplace=True)
city_df
#add lat long values of cities to the data of shoots
df_with_cities = pd.merge(df, city_df, on=["city","state"])
df_with_cities.head()
#count the cases in each county 
locations = df_with_cities.groupby(df["city"])
cases = locations["id"].count()
print(cases.max()," is maximum number of cases in a city and \n",cases.min()," is minimum number of cases of shooting")
cases
#map total number of cases with city
data = pd.merge(cases, city_df, on=["city"])
data = data.rename(columns = {"id":"count"})
data.head()
m = folium.Map(location=[32, -100], tiles='openstreetmap', zoom_start=3)

for idx, row in data.iterrows():
    Marker([row['latitude'], row['longitude']], popup=[row['city'],row["count"]]).add_to(m)
m
df_cluster = df_with_cities[["name","city","longitude","latitude"]]

m = folium.Map(location=[32, -100], tiles='openstreetmap', zoom_start=3)

mc = MarkerCluster()

for idx, row in df_cluster.iterrows():
    if not math.isnan(row['longitude']) and not math.isnan(row['latitude']):
        mc.add_child(folium.Marker([row['latitude'], row['longitude']], popup=[row['city'],row["name"]]))

m.add_child(mc)
m
m = folium.Map(location=[39, -119], tiles='cartodbpositron', zoom_start=4)

HeatMap(data=df_with_cities[['latitude', 'longitude']], radius=15).add_to(m)

m
states_full = gpd.read_file('/kaggle/input/us-administrative-boundaries/USA_adm1.shp')
states_geom = states_full[["NAME_1","geometry"]]
states_geom = states_geom.rename(columns={"NAME_1":"name"})
states_geom.head()
states = pd.read_csv("../input/states-geocoded-for-data-police-shootings/states.csv")
states.head()
state_count = df.groupby("state")
state_count = state_count["id"].count()
state_count = state_count.reset_index().rename(columns={"id":"count"})
print(state_count.max()["count"]," is the maximum number of incidents in a state and",state_count.min()["count"], " is the minimum.")
state_count.head()
#mapping state code with count of cases
df_with_states = pd.merge(states, state_count, on=["state"])
df_with_states = df_with_states.rename(columns={"id":"count"})
df_with_states.head()
#url to get data of the state boundaries of USA
url = 'https://raw.githubusercontent.com/python-visualization/folium/master/examples/data'
state_geo = f'{url}/us-states.json'
m = folium.Map(location=[48, -102], zoom_start=3)

folium.Choropleth(
    geo_data=state_geo,
    name='choropleth',
    data=state_count,
    columns=['state', 'count'],
    key_on='feature.id',
    fill_color='BuPu',
    fill_opacity=0.8,
    line_opacity=0.2,
    legend_name='Incidents '
).add_to(m)

folium.LayerControl().add_to(m)

m