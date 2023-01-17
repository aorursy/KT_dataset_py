# Importing packages 

import xarray as xr 
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
import geopandas as gpd 
import shapely
from shapely.geometry import Point
import datetime as dt
import numpy as np
from math import cos, asin, sqrt
from scipy.spatial.distance import cdist
%matplotlib inline 
pd.set_option('display.max_columns', None) 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('../input/cee-498-project12-earth-system-model/train.csv', parse_dates=['time'])
train
train.drop(columns=['area', 'date_written', 'landfrac', 'landmask', 'mcdate', 'mcsec', 'mdcur', 'mscur', 'nbedrock', 'nstep',
                   'pftmask', 'time_written'], axis=1, inplace=True)
train['time'] = pd.to_datetime(train['time'], errors='coerce') 
train.TSA.describe()
print(train[train.TSA == 0].time.min())
print(train[train.TSA == 0].time.max())
train.TSA.hist()
print(train[train.TSA < 5].time.min())
print(train[train.TSA < 5].time.max())
train = train[train.TSA > 173.15]
print(train.TSA.describe())
loc_mean = train.groupby(['lat','lon']).mean(); # spatial distribution of TSA over every time step for each location 
loc_mean.TSA.hist(bins=70)
data1 = {'Lat': train.lat,
         'Lon': train.lon}

data2 = {'Lat': 40.116421,
         'Lon': 271.756615}


def closest_point(point, points):
    """ Find closest point from a list of points. """
    return points[cdist([point], points).argmin()]

def match_value(df, col1, x, col2):
    """ Match value x from col1 row to value in col2. """
    return df[df[col1] == x][col2].values[0]


df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2, index=[0])

df1['point'] = [(x, y) for x,y in zip(df1['Lat'], df1['Lon'])]
df2['point'] = [(x, y) for x,y in zip(df2['Lat'], df2['Lon'])]

df2['closest'] = [closest_point(x, list(df1['point'])) for x in df2['point']]
print(df2)
champaign = train[(train.lon==271.25) & (train.lat==40.0523567199707)]
champaign.plot(figsize=(10,5), x='time', y='TSA', color='turquoise');
plt.title('Time Series of TSA near Champaign, IL');
plt.xlabel('Time');
plt.ylabel('Temperature (K)');
TSA_resamp_mean = train.set_index('time').resample('M')['TSA'].mean().dropna()
TSA_resamp_mean.plot(figsize=(15,5), color='turquoise');
plt.title('Time Series of Global Monthly TSA Mean');
plt.xlabel('Time');
plt.ylabel('Temperature (K)');
TSA_resamp_std = train.set_index('time').resample('M')['TSA'].std().dropna()
TSA_resamp_std.plot(figsize=(15,5), color='turquoise');
plt.title('Time Series of Global Monthly TSA Standard Deviation');
plt.xlabel('Time');
plt.ylabel('Temperature (K)');
TSA_resamp_std
TSA_corrs = train.corrwith(train.TSA).abs().sort_values(ascending=False).dropna()
TSA_corrs[TSA_corrs > 0.8].keys()
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world.crs
points = train.apply(lambda row: Point(row.lon, row.lat), axis=1)
gdf = gpd.GeoDataFrame(train, geometry=points)
gdf.time = pd.to_datetime(gdf.time)
tsa2015 = gdf[gdf.time.dt.year == 2015]
tsa2015_01 = tsa2015.TSA[tsa2015.time.dt.month == 2]
tsa2015_01 = gpd.GeoDataFrame(tsa2015_01, geometry=points)
tsa2015_01.set_crs(epsg=4326)

fig, ax = plt.subplots()
tsa2015_01.plot(ax=ax,
                column='TSA', 
                legend=True,
                figsize=(15,15),
                markersize=3,
               legend_kwds={'label': 'Temperature (K)',
                           'orientation': 'horizontal'})
plt.title('2m Air Temperature')
plt.show();
tsa2015_01.hist(bins=40);
tsa2015 = gdf[gdf.time.dt.year == 2015]
tsa2015_07 = tsa2015.TSA[tsa2015.time.dt.month == 8]
tsa2015_07 = gpd.GeoDataFrame(tsa2015_07, geometry=points)
tsa2015_07.set_crs(epsg=4326)

fig, ax = plt.subplots()
tsa2015_07.plot(ax=ax,
                column='TSA', 
                legend=True,
                figsize=(15,15),
                markersize=3,
               legend_kwds={'label': '2m Temperature',
                           'orientation': 'horizontal'})
plt.show();

tsa2015_07.hist(bins=40);
tv = gdf[gdf.time.dt.year == 2015]
tv = tv.TV[tv.time.dt.month == 2]
tv = gpd.GeoDataFrame(tv, geometry=points)
tv.set_crs(epsg=4326)

fig, ax = plt.subplots()
tv.plot(ax=ax,
                column='TV', 
                legend=True,
                figsize=(15,15),
                markersize=3,
               legend_kwds={'label': 'Temperature (K)',
                           'orientation': 'horizontal'})
plt.title('Vegetation Temperature')
plt.show();

tv.hist(bins=40);
th2osfc = gdf[gdf.time.dt.year == 2015]
th2osfc = th2osfc.TH2OSFC[th2osfc.time.dt.month == 2]
th2osfc = gpd.GeoDataFrame(th2osfc, geometry=points)
th2osfc.set_crs(epsg=4326)

fig, ax = plt.subplots()
th2osfc.plot(ax=ax,
                column='TH2OSFC', 
                legend=True,
                figsize=(15,15),
                markersize=3,
               legend_kwds={'label': 'Temperature (K)',
                           'orientation': 'horizontal'})
plt.title('Surface Water Temperature')
plt.show();

th2osfc.hist(bins=40);
tbuild = gdf[gdf.time.dt.year == 2015]
tbuild = tbuild.TBUILD[tbuild.time.dt.month == 2]
tbuild = gpd.GeoDataFrame(tbuild, geometry=points)
tbuild.set_crs(epsg=4326)

fig, ax = plt.subplots()
tbuild.plot(ax=ax,
                column='TBUILD', 
                legend=True,
                figsize=(15,15),
                markersize=3,
               legend_kwds={'label': 'Temperature (K)',
                           'orientation': 'horizontal'})
plt.title('Internal Urban Building Air Temperature')
plt.show();

tbuild.hist(bins=40);
var = gdf[gdf.time.dt.year == 2015]
var = var.ZWT_PERCH[var.time.dt.month == 2]
var = gpd.GeoDataFrame(var, geometry=points)
var.set_crs(epsg=4326)

fig, ax = plt.subplots()
var.plot(ax=ax,
                column='ZWT_PERCH', 
                legend=True,
                figsize=(15,15),
                markersize=3,
               legend_kwds={'label': 'mm',
                           'orientation': 'horizontal'})
plt.title('Perched Water Table Depth')
plt.show();

var.hist(bins=40);