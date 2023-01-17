import geopandas
import geoplot
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import sys
import altair as alt
#!pip install vega_datasets
from vega_datasets import data
from datetime import datetime
from datetime import date
from datetime import timedelta
import mapclassify
from shapely.ops import orient
dataset = pd.read_csv('/kaggle/input/corona-virus-report/covid_19_clean_complete.csv')
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
dataset.head()
dataset.shape
dataset.isnull().sum()
world.isnull().sum()
del dataset['Province/State']
del dataset['WHO Region']
del dataset['Lat']
del dataset['Long']
dataset.sort_values(by=['Date','Country/Region'], ascending=True,inplace=True)
for i in range(len(world)):
    if world.iloc[i]['name'] == 'United States of America':
        world.at[i,'name'] = 'US'
    if world.iloc[i]['name'] == 'Dem. Rep. Congo':
        world.at[i,'name'] = 'Congo'
for i in range(len(dataset)):
    if dataset.iloc[i]['Country/Region'] == 'Congo (Brazzaville)':
        dataset.at[i,'Country/Region'] = 'Congo'
    if dataset.iloc[i]['Country/Region'] == 'Congo (Kinshasa)':
        dataset.at[i,'Country/Region'] = 'Congo'
dataset = dataset.groupby(['Date','Country/Region']).sum()
dataset.reset_index(inplace=True)
dataset = pd.merge(dataset,world[['pop_est','name','gdp_md_est','geometry']],how='left',left_on='Country/Region',right_on='name')
dataset.dropna(inplace=True)
dataset[:10]
del dataset['Country/Region']
dataset = geopandas.GeoDataFrame(dataset, crs="EPSG:4326", geometry=dataset['geometry'])
temp = dataset[['geometry','name','Date','Confirmed']]
temp = temp[temp['Date']=='2020-01-22']
temp.geometry = temp.geometry.simplify(0.1)

temp.geometry = temp.geometry.apply(orient, args=(-1,))

# visz
alt.Chart(temp).mark_geoshape(
    stroke='black',
    strokeWidth=0.5
).encode(
    color=alt.Color('Confirmed:Q', scale=alt.Scale(domain=(0, 5),clamp=True)),
    #color='Confirmed:Q',
    tooltip=["name:N",'Confirmed:Q']
).transform_lookup(
    lookup='name',
    from_=alt.LookupData(temp, 'name', ['Confirmed'])
).properties(
    width=1000,
    height=600
)

temp = dataset[['geometry','name','Date','Deaths']]
temp = temp[temp['Date']=='2020-01-22']
temp.geometry = temp.geometry.simplify(0.1)

temp.geometry = temp.geometry.apply(orient, args=(-1,))

# visz
alt.Chart(temp).mark_geoshape(
    stroke='black',
    strokeWidth=0.5
).encode(
    color=alt.Color('Deaths:Q', scale=alt.Scale(domain=(0, 5),clamp=True)),
    #color='Confirmed:Q',
    tooltip=["name:N",'Deaths:Q']
).transform_lookup(
    lookup='name',
    from_=alt.LookupData(temp, 'name', ['Deaths'])
).properties(
    width=1000,
    height=600
)

temp = dataset[['geometry','name','Date','Confirmed']]
temp = temp[temp['Date']=='2020-07-27']
temp.geometry = temp.geometry.simplify(0.1)

temp.geometry = temp.geometry.apply(orient, args=(-1,))

# visz
alt.Chart(temp).mark_geoshape(
    stroke='black',
    strokeWidth=0.5
).encode(
    color=alt.Color('Confirmed:Q', scale=alt.Scale(domain=(0, 700000),clamp=True)),
    #color='Confirmed:Q',
    tooltip=["name:N",'Confirmed:Q']
).transform_lookup(
    lookup='name',
    from_=alt.LookupData(temp, 'name', ['Confirmed'])
).properties(
    width=1000,
    height=600
)

import mapclassify
deaths = dataset[dataset['Date']=='2020-01-22']['Confirmed']
scheme = mapclassify.Quantiles(deaths, k=50)

geoplot.choropleth(dataset[dataset['Date']=='2020-01-22']['geometry'], hue=deaths, scheme=scheme,cmap='Reds', figsize=(40, 15), legend=True)
deaths = dataset[dataset['Date']=='2020-07-26']['Confirmed']
scheme = mapclassify.Quantiles(deaths, k=50)

geoplot.choropleth(dataset[dataset['Date']=='2020-07-26']['geometry'], hue=deaths, scheme=scheme,cmap='Reds', figsize=(40, 15), legend=True)
