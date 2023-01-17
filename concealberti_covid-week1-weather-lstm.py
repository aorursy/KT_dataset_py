# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

try:
    tensorflow.__version__ >= "2.0"
except Exception:
    pass

import tensorflow as tf
from tensorflow import keras
# Common imports
# from pathlib import Path
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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_log_error

import os

# to make this notebook's output stable across runs
np.random.seed(42)
train = pd.read_csv( '/kaggle/input/covid19-global-forecasting-week-1/train.csv')
train=train.rename(columns={"Lat": "lat", "Long": "lon"})
mo = train['Date'].apply(lambda x: x[5:7])
da = train['Date'].apply(lambda x: x[8:10])
train['day_from_jan_first'] = (da.apply(int)
                               + 31*(mo=='02') 
                               + 60*(mo=='03')
                               + 91*(mo=='04')  
                              )

train= train.dropna(subset = ['lat', 'lon'])
train = train.reset_index(drop=True)
train.lon= train.lon.astype(int)
train.lat= train.lat.astype(int)
#Realized that locations with same latitude, longitude could be located in 2 different countries
train=train.rename(columns={'Country/Region': 'Country'})
columns_train = ["Country", "lat", "lon"]
countLONLATCountry= train.groupby(columns_train, as_index=False).agg({'Id': np.count_nonzero, 'ConfirmedCases': np.mean})
#infact at -122, 37 there are 126 datapoints instead of 63
countLONLAT= train['Id'].groupby([train['lon'], train['lat']]).agg(['count'])

# with lat, lon AND country they are univoquely identified, except for 1 location
countLONLATCountry= train.groupby(columns_train, as_index=False).agg(['count'])
countLONLATCountry[countLONLATCountry> 63].agg(['count'])
(countLONLATCountry > 63).agg('sum')
#that location is 
countLONLATCountry= train.groupby(columns_train, as_index=False).agg({'Id': np.count_nonzero, 'ConfirmedCases': np.mean})
countLONLATCountry[(countLONLATCountry['Id'] > 63)==True]
#Create a New dataset without that location: this is the easiest to have different countries with same number of timesteps
train_new= train[train.lon !=-64]
train_new.shape
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
weather_df= weather_df.dropna(subset = ['lat', 'lon'])
weather_df = weather_df.reset_index(drop=True)
weather_df.lon= weather_df.lon.astype(int)
weather_df.lat= weather_df.lat.astype(int)

CovidWeatherTrain=train_new.merge(weather_df, how='left', on=['lat', 'lon', 'day_from_jan_first'])
# list(CovidWeatherTrain.columns)
#TO PLOT COORDINATES THE COORD NEED TO BE TRANSFORMED BACK TO INT

CovidWeatherTrain['lon']= CovidWeatherTrain['lon'].astype(int) 
CovidWeatherTrain['lat']= CovidWeatherTrain['lat'].astype(int)

CovidWeather=CovidWeatherTrain
# There is a lot of missing data in dew point
geom= [Point(xy) for xy in zip(CovidWeather['lon'], CovidWeather['lat'])]
crs={'init': 'epsg:4326'}
geo_df= gpd.GeoDataFrame(CovidWeather, crs=crs, geometry= geom)
fig, ax= plt.subplots(figsize = (15,15))
geo_df.plot(ax= ax, markersize=CovidWeather.dewp, marker= "o")
#TO DO GROUPBY THE COLUMNS NEED TO BE STR
CovidWeatherTrain['lon']= CovidWeatherTrain['lon'].astype(str) 
CovidWeatherTrain['lat']= CovidWeatherTrain['lat'].astype(str)
CovidWeatherTrain['Location'] = CovidWeatherTrain['lon'] + '-' + CovidWeatherTrain['lat'] + '-' + CovidWeatherTrain['Country']
CovidWeatherTrain['Location'][1]

columns_train= ["Country", "lat", "lon" , 'day_from_jan_first', "Location"]
CovidWeather=CovidWeatherTrain

columns_Aggregate= ["Location", "day_from_jan_first" ]

CovidWeatherAggregated= CovidWeather.groupby(columns_Aggregate, as_index=False).agg({ 'dewp': np.nanmean, "Fatalities": np.nanmean, "ConfirmedCases": np.nanmean}) 
CovidWeatherAggregated = CovidWeatherAggregated.dropna() 
len(CovidWeatherAggregated)
columns_X = ["Location", "day_from_jan_first", "dewp" ]
columns_y_Cases= [ "Location", "day_from_jan_first", "ConfirmedCases"]
columns_y_Fatalities= [ "Location", "day_from_jan_first", "Fatalities" ]

CovidWeatherAggregated.data= CovidWeatherAggregated[columns_X]
CovidWeatherAggregated.target1= CovidWeatherAggregated[columns_y_Cases]
CovidWeatherAggregated.target2= CovidWeatherAggregated[columns_y_Fatalities]
# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)
def model_build(Pivoted):
    Train, Test  = Pivoted[:round(len(Pivoted)/2)], Pivoted[round(len(Pivoted)/2):]
    ArrayTrain, ArrayTest = np.array(Train), np.array(Test)
    
    #in time series that are not cyclical (like Fatalities and Confirmed Cases in our data) 
    #we cannot split the data in train and test in cronological order, because the underlying function is 
    #exponential
    
#     Array = np.array(Pivoted)
#     tscv = TimeSeriesSplit()
#     TimeSeriesSplit(max_train_size=None, n_splits=9)
#     for train_index, test_index in tscv.split(Array):
#         print("TRAIN:", train_index, "TEST:", test_index)
#         ArrayTrain, ArrayTest = Array[train_index], Array[test_index]

    imputer = SimpleImputer(missing_values= np.nan, strategy='mean')
    imputer=imputer.fit(ArrayTrain)
    Train_imputed = imputer.transform(ArrayTrain)
    #this subsitite the nan with the average per column (from previuos or following days. separately for each country)

    imputer = SimpleImputer(missing_values= np.nan, strategy='mean')
    imputer=imputer.fit(ArrayTest)
    Test_imputed = imputer.transform(ArrayTest)
    
    scaler = StandardScaler()
    Train_scaled = scaler.fit_transform(Train_imputed)
    Test_scaled = scaler.transform(Test_imputed)
    
    n_steps=3
    Xtrain, ytrain = split_sequences(Train_imputed, n_steps)
    Xtest, ytest = split_sequences(Test_imputed, n_steps)
    
    n_features= Xtrain.shape[2]
    
    model =  keras.models.Sequential([
    #Return_sequence tells whether to return the last output in the output sequence or the full sequence
    #True is used to return the hidden state output for each input time step.
    #if Stacking LSTM Return_sequence must be set to True
    keras.layers.LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)),
    keras.layers.LSTM(100, activation='relu'),
    #the number of neurons in the Dense layer is one because we want a single value per country
    keras.layers.Dense(n_features)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    
    history= model.fit(Xtrain, ytrain, epochs= 30)
    
    yhat = model.predict(Xtest, verbose=0)
    
    Columns= list(Pivoted.columns)
    

    Test_reset= Test.reset_index()
    TestReduced= Test_reset[1:len(Test)-1]
    Test_mean= TestReduced.mean(axis=1)

    yhat_df= pd.DataFrame(yhat, columns=  Columns)
    yhat_df['day_from_jan_first']= Test_reset['day_from_jan_first']
    yhat_mean= yhat_df.mean(axis=1)

    fig = plt.figure()
    fig.suptitle('Prediction and Loss', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(1,2,1)
    yhat_mean.plot(ax= ax, x='day_from_jan_first', y= Columns, color='red', label= 'predicted')
    Test_mean.plot(ax= ax, x='day_from_jan_first', y= Columns, color='black', label= 'real values')
    ax.set_xlabel('days')
    ax.set_title('Prediction')
    ax.legend(loc="upper right")
    
    ax2 = fig.add_subplot(1,2,2)
    loss= history.history['loss']
    epochs=range(len(loss))
    ax2.plot(epochs, loss, 'bo', label='Training Loss')
    ax2.set_xlabel('epochs')
    ax2.set_title('Training Loss')
    plt.show()
    
    return Train, Test, yhat_df
CovidWeatherXPivoted = CovidWeatherAggregated.data.pivot(index='day_from_jan_first',columns='Location',values='dewp')
Xtrain1, Xtest1, yhat_df1= model_build(CovidWeatherXPivoted)
CovidWeatherYCasesPivoted = CovidWeatherAggregated.target1.pivot(index='day_from_jan_first',columns='Location',values='ConfirmedCases')
Xtrain2, Xtest2, yhat_df2= model_build(CovidWeatherYCasesPivoted)
CovidWeatherYFatalitiesPivoted = CovidWeatherAggregated.target2.pivot(index='day_from_jan_first',columns='Location',values='Fatalities')
Xtrain3, Xtest3, yhat_df3= model_build(CovidWeatherYFatalitiesPivoted)