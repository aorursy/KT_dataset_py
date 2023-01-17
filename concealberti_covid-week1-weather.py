# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import json
from pathlib import Path
from google.cloud import bigquery

import matplotlib.pyplot as plt
from matplotlib import colors

import seaborn as sns

import descartes
import geopandas as gpd
from shapely.geometry import Point, Polygon

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
train = pd.read_csv( '/kaggle/input/covid19-global-forecasting-week-1/train.csv')
train=train.rename(columns={"Lat": "lat", "Long": "lon"})
train
train['Id'].groupby(train["Country/Region"]).agg(['count'])
mo = train['Date'].apply(lambda x: x[5:7])
da = train['Date'].apply(lambda x: x[8:10])
train['day_from_jan_first'] = (da.apply(int)
                               + 31*(mo=='02') 
                               + 60*(mo=='03')
                               + 91*(mo=='04')  
                              )

train=train.rename(columns={'Country/Region': 'Country'})
list(train.Country[(train.day_from_jan_first==80)])
# train[(train.day_from_jan_first==80)]
geom= [Point(xy) for xy in zip(train['lon'], train['lat'])]

crs={'init': 'epsg:4326'}
geo_df= gpd.GeoDataFrame(train, crs=crs, geometry= geom)
fig, ax= plt.subplots(figsize = (15,15))
geo_df.plot(ax= ax, markersize=20, marker= "o")
# Set your own project id here
PROJECT_ID = 'your-google-cloud-project'
from google.cloud import bigquery
client = bigquery.Client(project=PROJECT_ID)
table1_stations = bigquery.TableReference.from_string(
    "bigquery-public-data.noaa_gsod.stations"
)

dataframe_stations = client.list_rows(
    table1_stations,
    selected_fields=[
        bigquery.SchemaField("usaf", "STRING"), #station number, world metherorological org
        bigquery.SchemaField("wban", "STRING"), #wban number, weather bureau army
        bigquery.SchemaField("country", "STRING"),
        bigquery.SchemaField("lat", "FLOAT"),
        bigquery.SchemaField("lon", "FLOAT"),
    ],
).to_dataframe()

dataframe_stations
table1_gsod2020 = bigquery.TableReference.from_string(
    "bigquery-public-data.noaa_gsod.gsod2020"
)

dataframe_gsod2020= client.list_rows(table1_gsod2020,
    selected_fields=[
        bigquery.SchemaField("stn", "STRING"), #station number
        bigquery.SchemaField("wban", "STRING"), #station number
        bigquery.SchemaField("year", "INTEGER"),
        bigquery.SchemaField("mo", "INTEGER"),
        bigquery.SchemaField("da", "INTEGER"),
        bigquery.SchemaField("temp", "FLOAT"), #mean temp of the day
        bigquery.SchemaField("dewp", "FLOAT"), #mean_dew_point
        bigquery.SchemaField("slp", "FLOAT"), #mean_sealevel_pressure
        bigquery.SchemaField("wdsp", "FLOAT"), #mean_wind_speed
        bigquery.SchemaField("prcp", "FLOAT"), #total_precipitation
        bigquery.SchemaField("sndp", "FLOAT"), #snow_depth
    ],).to_dataframe()

dataframe_gsod2020
stations_df= dataframe_stations
twenty_twenty_df= dataframe_gsod2020
stations_df['STN'] = stations_df['usaf'] + '-' + stations_df['wban']
twenty_twenty_df['STN'] = twenty_twenty_df['stn'] + '-' + twenty_twenty_df['wban']
cols_1= list(twenty_twenty_df.columns)
cols_2= list(stations_df.columns)
weather_df = twenty_twenty_df[cols_1].join(stations_df[cols_2].set_index('STN'), on='STN',  how='left', lsuffix='_left', rsuffix='_right')

weather_df['temp'] = weather_df['temp'].apply(lambda x: np.nan if x==9999.9 else x)
weather_df['slp'] = weather_df['slp'].apply(lambda x: np.nan if x==9999.9 else x)
weather_df['dewp'] = weather_df['dewp'].apply(lambda x: np.nan if x==9999.9 else x)
weather_df['wdsp'] = weather_df['wdsp'].apply(lambda x: np.nan if x==999.9 else x)
weather_df['prcp'] = weather_df['prcp'].apply(lambda x: np.nan if x==999.9 else x)
weather_df['sndp'] = weather_df['sndp'].apply(lambda x: np.nan if x==999.9 else x)

# convert everything into celsius
temp = (weather_df['temp'] - 32) / 1.8
dewp = (weather_df['dewp'] - 32) / 1.8
    
# compute relative humidity as ratio between actual vapour pressure (computed from dewpoint temperature)
# and saturation vapour pressure (computed from temperature) (the constant 6.1121 cancels out)
weather_df['rh'] = (np.exp((18.678*dewp)/(257.14+dewp))/np.exp((18.678*temp)/(257.14+temp)))

# calculate actual vapour pressure (in pascals)
# then use it to compute absolute humidity from the gas law of vapour 
# (ah = mass / volume = pressure / (constant * temperature))
weather_df['ah'] = ((np.exp((18.678*dewp)/(257.14+dewp))) * 6.1121 * 100) / (461.5 * temp)


weather_df['month']= weather_df['mo']
weather_df['day']= weather_df['da']
weather_df['Date']=pd.to_datetime(weather_df[['year','month','day']])
weather_df['Date2']= weather_df['Date']
weather_df['Date2']= weather_df['Date2'].astype('str')
mo2 = weather_df['Date2'].apply(lambda x: x[5:7])
da2 = weather_df['Date2'].apply(lambda x: x[8:10])
weather_df['day_from_jan_first'] = (da2.apply(int)
                               + 31*(mo2=='02') 
                               + 60*(mo2=='03')
                               + 91*(mo2=='04')  
                              )

geom= [Point(xy) for xy in zip(weather_df['lon'], weather_df['lat'])]
crs={'init': 'epsg:4326'}
geo_df= gpd.GeoDataFrame(weather_df, crs=crs, geometry= geom)
fig, ax= plt.subplots(figsize = (15,15))
geo_df.plot(ax= ax, markersize=20, marker= "o")
weather_df= weather_df.dropna(subset = ['lat', 'lon'])
weather_df = weather_df.reset_index(drop=True)
train= train.dropna(subset = ['lat', 'lon'])
train = train.reset_index(drop=True)
weather_df.lon= weather_df.lon.astype(int)
weather_df.lat= weather_df.lat.astype(int)
train.lon= train.lon.astype(int)
train.lat= train.lat.astype(int)

CovidWeather=train.merge(weather_df, on=['lat', 'lon', 'day_from_jan_first'], how='left')
CovidWeather
columns_X = ["lat", "lon","temp", "dewp", "slp", "wdsp", "prcp", "sndp", "rh", "ah"]
columns_y= [ "ConfirmedCases", "Fatalities" ]


weather_PerDay2=CovidWeather[["lat", "lon","temp", "dewp", "slp", "wdsp", "prcp", "sndp", "rh", "ah", "ConfirmedCases", "Fatalities"]]
weather_PerDay2.replace([np.inf, -np.inf], np.nan, inplace=True)

weather_PerDay2.data= CovidWeather[columns_X]
weather_PerDay2.target= CovidWeather[columns_y]
X_train_full, X_test, y_train_full, y_test = train_test_split(
    weather_PerDay2.data, weather_PerDay2.target, random_state=42)

X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
X_train_full.replace([np.inf, -np.inf], np.nan, inplace=True)

imputer = SimpleImputer(missing_values= np.nan, strategy='mean')
imputer=imputer.fit(X_train_full)
X_train_full = imputer.transform(X_train_full)
X_train_full

imputer = SimpleImputer(missing_values= np.nan, strategy='mean')
imputer=imputer.fit(X_test)
X_test = imputer.transform(X_test)
X_test

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test)


rnd_reg= RandomForestRegressor()
rnd_reg.fit (X_train_scaled, y_train_full)

importances = list(rnd_reg.feature_importances_)
importances

feature_list = list(weather_PerDay2.data.columns)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
feature_importances
y_pred= rnd_reg.predict(X_test)
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)
plt.scatter(weather_PerDay2.dewp,weather_PerDay2.ConfirmedCases)
plt.show()
plt.scatter(weather_PerDay2.dewp,weather_PerDay2.Fatalities)
plt.show()
Selectdf= weather_df[(weather_df['dewp'].apply(lambda x:x>=30 and x<=60))]
# Selectdwep
crs={'init': 'epsg:4326'}
geom_Selectdf= [Point(xy) for xy in zip(Selectdf['lon'], Selectdf['lat'])]
geo_Selectdf= gpd.GeoDataFrame(Selectdf, crs=crs, geometry= geom_Selectdf) #the one with critical dewpoint

geom= [Point(xy) for xy in zip(weather_df['lon'], weather_df['lat'])]
geo_df= gpd.GeoDataFrame(weather_df, crs=crs, geometry= geom)

fig, ax= plt.subplots(figsize = (15,15))
geo_df.plot(ax= ax, markersize=20, color= 'b', marker= "o")
geo_Selectdf.plot(ax= ax, markersize=20, color= 'r', marker= "o")


Selectdewp= weather_PerDay2[(weather_PerDay2['dewp'].apply(lambda x:x>=30 and x<=60))]
# Selectdwep

geom_Selectdewp= [Point(xy) for xy in zip(Selectdewp['lon'], Selectdewp['lat'])]
geo_Selectdewp= gpd.GeoDataFrame(Selectdewp, crs=crs, geometry= geom_Selectdewp) #the one with critical dewpoint

geom= [Point(xy) for xy in zip(weather_PerDay2['lon'], weather_PerDay2['lat'])]
geo_df= gpd.GeoDataFrame(weather_PerDay2, crs=crs, geometry= geom) #all dataset


fig, ax= plt.subplots(figsize = (15,15))
geo_df.plot(ax= ax, markersize=20, color= 'b', marker= "o")
geo_Selectdewp.plot(ax= ax, markersize=20, color= 'r', marker= "o")
param_grid = [{ "n_estimators": [9, 10] , "max_features" : [7, 10]}]
        
score='neg_mean_squared_error'
   
classifier= RandomForestRegressor()
gridsearch = GridSearchCV(classifier,param_grid, scoring = score, cv = 5)
my_model= gridsearch.fit(X_train_scaled, y_train_full)
cv_results= gridsearch.cv_results_
for mean_score, params in zip(cv_results["mean_test_score"], cv_results["params"]):
    print(np.sqrt(-mean_score), params)

y_pred= my_model.predict(X_test)

from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)
    