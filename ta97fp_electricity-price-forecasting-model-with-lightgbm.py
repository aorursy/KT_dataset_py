# Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from pylab import rcParams
import seaborn as sn
from scipy import stats
from geopy.geocoders import Nominatim
import plotly as pl
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from functools import reduce
import requests
import json
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from yellowbrick.features import Rank1D
from yellowbrick.features import Rank2D
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import lightgbm as lgb
from sklearn.metrics import r2_score
from sklearn import metrics
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LassoLarsCV
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import PolynomialFeatures
#Set numpy output options 

np.set_printoptions(edgeitems=3)
np.core.arrayprint._line_width = 30

# Pandas output options
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 100)
%time

# Data
energyData = pd.read_csv('/kaggle/input/energy-consumption-generation-prices-and-weather/energy_dataset.csv')
weatherData = pd.read_csv('/kaggle/input/energy-consumption-generation-prices-and-weather/weather_features.csv')

# Set time as index
energyData.set_index('time', inplace = True)
weatherData.set_index('dt_iso', inplace = True)
# Energy Data Overview

energyData.head(15)
# Rename all columns

energyData.columns = energyData.columns.map(lambda x : x+'_MWh' if x !='price day ahead' and x!='price actual' else x)

columns = energyData.columns[energyData.columns.str.contains('price day ahead|price actual')]
energyData.rename(columns = dict(zip(columns, columns + '_€/Mwh')), inplace=True)


# Check all nan values

print('Energy Data NaN values: \n', energyData.isna().sum())
# Check the features that have the same number of NaN values as the lenght of the dataframe

def CheckNull(data_frame):
    for i in data_frame.columns.values:
        if data_frame[i].isna().sum() == len(data_frame):
            print('This column is empty: ', i)
            
CheckNull(energyData)
# Dropping the NaN columns
energyData = energyData.drop(['generation hydro pumped storage aggregated_MWh', 'forecast wind offshore eday ahead_MWh'], axis=1)

# Substituting NaN values in energy dataset with linear interpolation
energyData.interpolate(method='linear', inplace=True, axis=0)

# Checking for duplicated values

duplicatedEnergy_values = energyData.duplicated().sum()
print('There is {} duplicated values.'.format(duplicatedEnergy_values))
# Check dataset dtype

print('Dataset Type \n', energyData.dtypes)
# Checking for the values distribution for more cleaning

energyData.hist(figsize=(25, 30), bins=50, xlabelsize=10, ylabelsize=10)
plt.show()
# Dropping least relevant columns

energyData = energyData.drop(['generation fossil coal-derived gas_MWh', 'generation fossil oil shale_MWh', 'generation fossil peat_MWh', 'generation geothermal_MWh', 'generation marine_MWh', 'generation wind offshore_MWh'], axis=1)
# Weather data overview

weatherData.head(15)
# Verifying NaN values at each column

if CheckNull(weatherData) == None:
    print('All collumns have values')
    
print('Weather Data: \n', weatherData.isna().sum())
# Check dataset type 
print('Dataset Type \n', weatherData.dtypes)
#Column values distribution

weatherData.hist(figsize=(25, 30), bins=50, xlabelsize=10, ylabelsize=10)
plt.show()
cities = weatherData['city_name'].unique().tolist()

print('Weather Cities: \n', cities)
# Defining our locator function
geolocator = Nominatim()

# Function for latitude and longitude information
def geo_locator(city, country):
    loc = geolocator.geocode(str(city + ',' + country))
    return (loc.latitude, loc.longitude)

# Coordinates
latitudes = []
longitudes = []

# Geolocate from city list
for i in cities:
    location = geo_locator(i,'Spain')
    latitudes.append(location[0])
    longitudes.append(location[1])
    
    
weatherData['Latitude'] = 0
weatherData['Longitude'] = 0

# Filling latitude and longitude for each city

weatherData['Latitude'].loc[weatherData['city_name']=='Valencia'] = latitudes[0]
weatherData['Latitude'].loc[weatherData['city_name']=='Madrid'] = latitudes[1]
weatherData['Latitude'].loc[weatherData['city_name']=='Bilbao'] = latitudes[2]
weatherData['Latitude'].loc[weatherData['city_name']==' Barcelona'] = latitudes[3]
weatherData['Latitude'].loc[weatherData['city_name']=='Seville'] = latitudes[4]

weatherData['Longitude'].loc[weatherData['city_name']=='Valencia'] = longitudes[0]
weatherData['Longitude'].loc[weatherData['city_name']=='Madrid'] = longitudes[1]
weatherData['Longitude'].loc[weatherData['city_name']=='Bilbao'] = longitudes[2]
weatherData['Longitude'].loc[weatherData['city_name']==' Barcelona'] = longitudes[3]
weatherData['Longitude'].loc[weatherData['city_name']=='Seville'] = longitudes[4]
# Checking unique values in categorical features

weatherMain_values = weatherData['weather_main'].unique().tolist()
weatherDescription_values = weatherData['weather_description'].unique().tolist()
weatherIcon_values = weatherData['weather_icon'].unique().tolist()

print('Weather Main unique values: \n', weatherMain_values)
print('Weather Description unique values: \n', weatherDescription_values)
print('Weather Icon unique values: \n', weatherIcon_values)


# Setting Label Encoder for categorical values

label_encoder = LabelEncoder()

weatherData['weather_main'] = label_encoder.fit_transform(weatherData['weather_main'])
weatherData['weather_description'] = label_encoder.fit_transform(weatherData['weather_description'])
weatherData['weather_icon'] = label_encoder.fit_transform(weatherData['weather_icon'])
print('Energy Data Lenght:', len(energyData))
print('Weather Data Lenght:', len(weatherData))

if (len(energyData) != (len(weatherData)/5)):
    print('There are duplicate values in weather data')
    
    
duplicatedWeather_values = weatherData.duplicated().sum()
print('There are {} duplicated values.'.format(duplicatedWeather_values))

# Dropping duplicated values by city and time
weatherData = weatherData.reset_index().drop_duplicates(subset=['dt_iso', 'city_name']).set_index('dt_iso')

# Renaming index
weatherData = weatherData.reset_index()
weatherData = weatherData.rename(columns = {'dt_iso':'time'})
#Subdividing weather information by city
weatherData1, weatherData2, weatherData3, weatherData4, weatherData5 = [y for _, y in weatherData.groupby('city_name')]

# Add sufix to each feature function
def addcity(dataframe):
    city_name = dataframe.iloc[0]['city_name']
    dataframe = dataframe.set_index(['time'])
    dataframe = dataframe.drop(['city_name'], axis = 1)
    dataframe = dataframe.add_suffix(city_name)
    return dataframe

weatherData_list = [weatherData1, weatherData2, weatherData3, weatherData4, weatherData5]

weatherData_result = []

# Applying the function to all weather data sets
for i in weatherData_list:
    weatherData_result.append(addcity(i))
    
# For merging purposes   
energyData = energyData.reset_index()

# For merging purposes
for i in range(0, len(weatherData_result)):
    weatherData_result[i] = weatherData_result[i].reset_index()
    
    
# Joining weather and energy data
completeDataset = reduce(lambda x,y: pd.merge(x,y, on='time'), [energyData, weatherData_result[0], weatherData_result[1], weatherData_result[2], weatherData_result[3], weatherData_result[4]])
# Complete dataset

completeDataset.head(15)
# Plot parameters
rcParams['figure.figsize'] = 10, 5

# Seaborn boxplot 
sn.boxplot(x=completeDataset['price actual_€/Mwh'])
plt.title('Dataset Outliers')
plt.show()
# Defining Z_score
z = np.abs(stats.zscore(completeDataset['price actual_€/Mwh']))

# Removing outliers
completeDataset = completeDataset[(z < 3)]

sn.boxplot(x=completeDataset['price actual_€/Mwh'])
plt.title('Dataset Outliers after first removal')
plt.show()

print('Lenght after 1st removal:', len(completeDataset))
# Defining the quantile IQR

Q1 = completeDataset['price actual_€/Mwh'].quantile(0.25)
Q3 = completeDataset['price actual_€/Mwh'].quantile(0.75)
IQR = Q3 - Q1

completeDataset = completeDataset[~((completeDataset['price actual_€/Mwh'] < (Q1 - 1.5 * IQR)) | (completeDataset['price actual_€/Mwh'] > (Q3 + 1.5 * IQR)))]

sn.boxplot(x=completeDataset['price actual_€/Mwh'])
plt.title('Dataset Outliers after second removal')
plt.show()

print('Lenght after 2nd removal:', len(completeDataset))
completeDataset.reset_index(drop=True)

# Taking format
completeDataset['time'] = completeDataset['time'].str[:-9]

completeDataset['time'] = completeDataset['time'].apply(lambda x: pd.to_datetime(str(x), format='%Y-%m-%d %H:%M'))
completeDataset['time'] = completeDataset['time'].dt.strftime('%d-%m-%Y %H:%M')
completeDataset['time'] = completeDataset['time'].apply(lambda x: pd.to_datetime(str(x), format='%d-%m-%Y %H:%M'))

# Getting date feature
completeDataset['date'] = completeDataset['time'].dt.date


# Getting hour_minute feature
completeDataset['Hour_Minute'] = completeDataset['time'].dt.time


# Week day
completeDataset['Week_Day'] = completeDataset['time'].dt.weekday
# General Statistics

print('Data Statistics \n', completeDataset.describe())

print(completeDataset.isna().sum())
# Figure subplot size
fig = plt.figure(figsize=(15,13))
ax = fig.add_subplot(111)


completeDataset = completeDataset.set_index('time', drop = False)

# Weekend example
startDate = '2015-01-01 00:00:00'
endDate = '2015-03-31 00:00:00'




ax.plot(completeDataset['date'][startDate:endDate],completeDataset['generation biomass_MWh'][startDate:endDate], color='r', label='biomass')
ax.plot(completeDataset['date'][startDate:endDate],completeDataset['generation fossil brown coal/lignite_MWh'][startDate:endDate], color='g', label='fossil brown coal/lignite')
ax.plot(completeDataset['date'][startDate:endDate],completeDataset['generation fossil gas_MWh'][startDate:endDate], color='grey', label='g')
ax.plot(completeDataset['date'][startDate:endDate],completeDataset['generation fossil hard coal_MWh'][startDate:endDate], color='y', label='hard coal')
ax.plot(completeDataset['date'][startDate:endDate],completeDataset['generation fossil oil_MWh'][startDate:endDate], color='c', label='oil')
ax.plot(completeDataset['date'][startDate:endDate],completeDataset['generation hydro pumped storage consumption_MWh'][startDate:endDate], color='m', label='hydro pumped storage')


plt.legend(loc='upper right')
plt.title('Energy sources contribution 1')
plt.xlabel('Date')
plt.ylabel('MW/h')
plt.show()
fig = plt.figure(figsize=(15,13))
ax = fig.add_subplot(111)

completeDataset = completeDataset.set_index('time', drop = False)

# Weekend example
startDate = '2015-01-01 00:00:00'
endDate = '2015-03-31 00:00:00'


ax.plot(completeDataset['date'][startDate:endDate],completeDataset['generation hydro run-of-river and poundage_MWh'][startDate:endDate], color='r', label='hydro run-of-river and poundage')
ax.plot(completeDataset['date'][startDate:endDate],completeDataset['generation hydro water reservoir_MWh'][startDate:endDate], color='g', label='hydro water reservoir')
ax.plot(completeDataset['date'][startDate:endDate],completeDataset['generation nuclear_MWh'][startDate:endDate], color='grey', label='nuclear')
ax.plot(completeDataset['date'][startDate:endDate],completeDataset['generation other_MWh'][startDate:endDate], color='y', label='other')
ax.plot(completeDataset['date'][startDate:endDate],completeDataset['generation other renewable_MWh'][startDate:endDate], color='c', label='other renewable')
ax.plot(completeDataset['date'][startDate:endDate],completeDataset['generation solar_MWh'][startDate:endDate], color='m', label='solar')

plt.legend(loc='upper right')
plt.title('Energy sources contribution 2')
plt.xlabel('Date')
plt.ylabel('MW/h')
plt.show()

# Figure subplot size
fig = plt.figure(figsize=(15,13))
ax = fig.add_subplot(111)

ax.plot(completeDataset['date'][startDate:endDate],completeDataset['generation waste_MWh'][startDate:endDate], color='r', label='w')
ax.plot(completeDataset['date'][startDate:endDate],completeDataset['generation wind onshore_MWh'][startDate:endDate], color='g', label='wind')

plt.legend(loc='upper right')
plt.title('Energy sources contribution 3')
plt.xlabel('Date')
plt.ylabel('MW/h')
plt.show()
# Figure subplot size
fig = plt.figure(figsize=(20,14))
ax = fig.add_subplot(111)

aset = '2015-01-01 00:00:00'
abset = '2015-12-31 00:00:00'

acset = '2016-01-01 00:00:00'
adset = '2016-12-31 00:00:00'

aeset = '2017-01-01 00:00:00'
afset = '2017-12-31 00:00:00'

agset = '2018-01-01 00:00:00'
ahset = '2018-12-31 00:00:00'




ax.plot(completeDataset['date'][aset:abset] ,completeDataset['total load actual_MWh'][aset:abset], color='c', label = '2015')
ax.plot(completeDataset['date'][acset:adset] ,completeDataset['total load actual_MWh'][acset:adset], color='r', label = '2016')
ax.plot(completeDataset['date'][aeset:afset] ,completeDataset['total load actual_MWh'][aeset:afset], color='b', label = '2017')
ax.plot(completeDataset['date'][agset:ahset] ,completeDataset['total load actual_MWh'][agset:ahset], color='g', label = '2018')


plt.legend(loc='upper right')
plt.title('Energy demand at each year')
plt.xlabel('Date')
plt.ylabel('MW/h')
plt.show()
energy_metrics = ['total load actual_MWh', 'price actual_€/Mwh']

weather_metrics = completeDataset.loc[:, 'temp Barcelona':'LongitudeValencia']

weather_metrics = weather_metrics.drop(['LatitudeBilbao', 'LongitudeBilbao', 'LatitudeValencia', 'LongitudeValencia', 'LatitudeMadrid', 'LongitudeMadrid', 'Latitude Barcelona', 'Longitude Barcelona', 'LatitudeSeville', 'LongitudeSeville'], axis=1)

cont = pd.merge(completeDataset[energy_metrics], weather_metrics, left_index=True, right_index=True)

# Correlation Matrix

calculation = cont.corr()


print('Energy matrix \n', calculation['total load actual_MWh'])

print('Price matrix \n', calculation['price actual_€/Mwh'])
# Price distribution

sn.distplot(completeDataset['price actual_€/Mwh'])
plt.title('Price Distribution')
plt.show()
# Figure subplot size
fig = plt.figure(figsize=(20,14))
ax = fig.add_subplot(111)

aset = '2015-01-01 00:00:00'
abset = '2015-12-31 00:00:00'

acset = '2016-01-01 00:00:00'
adset = '2016-12-31 00:00:00'

aeset = '2017-01-01 00:00:00'
afset = '2017-12-31 00:00:00'

agset = '2018-01-01 00:00:00'
ahset = '2018-12-31 00:00:00'




ax.plot(completeDataset['date'][aset:abset] ,completeDataset['price actual_€/Mwh'][aset:abset], color='c', label = '2015')
ax.plot(completeDataset['date'][acset:adset] ,completeDataset['price actual_€/Mwh'][acset:adset], color='r', label = '2016')
ax.plot(completeDataset['date'][aeset:afset] ,completeDataset['price actual_€/Mwh'][aeset:afset], color='b', label = '2017')
ax.plot(completeDataset['date'][agset:ahset] ,completeDataset['price actual_€/Mwh'][agset:ahset], color='g', label = '2018')


plt.legend(loc='upper right')
plt.title('Energy demand at each year')
plt.xlabel('Date')
plt.ylabel('MW/h')
plt.show()
# Demand Weight Feature

Bilbao_weight = 1
Seville_weight = 2
Valencia_weight = 3
Barcelona_weight = 4
Madrid_weight = 5


completeDataset['Bilbao_weight'] = Bilbao_weight
completeDataset['Seville_weight'] = Seville_weight
completeDataset['Valencia_weight'] = Valencia_weight
completeDataset['Barcelona_weight'] = Barcelona_weight
completeDataset['Madrid_weight'] = Madrid_weight
completeDataset['coal_oil_fossil_MWh'] = completeDataset['generation fossil brown coal/lignite_MWh'] + completeDataset['generation fossil gas_MWh'] + completeDataset['generation fossil hard coal_MWh'] + completeDataset['generation fossil oil_MWh']

completeDataset['renewables_MWh'] = completeDataset['generation hydro pumped storage consumption_MWh'] + completeDataset['generation hydro run-of-river and poundage_MWh'] + completeDataset['generation hydro water reservoir_MWh'] + completeDataset['generation other renewable_MWh'] + completeDataset['generation solar_MWh'] + completeDataset['generation wind onshore_MWh']
set_value = 0.05

weather_features = []

for index, value in calculation['price actual_€/Mwh'].items():
    if value > 0.05:
        weather_features.append(index)
        
print('Relevant Features: \n', weather_features)
relevant_features = ['date', 'Hour_Minute', 'Week_Day', 'total load forecast_MWh', 'total load actual_MWh', 'price day ahead_€/Mwh', 'price actual_€/Mwh', 'LatitudeBilbao', 'LongitudeBilbao', 'LatitudeValencia', 'LongitudeValencia', 'LatitudeMadrid', 'LongitudeMadrid', 'Latitude Barcelona', 'Longitude Barcelona', 'LatitudeSeville', 'LongitudeSeville', 'temp Barcelona', 'temp_min Barcelona', 'temp_max Barcelona', 'weather_description Barcelona', 'tempBilbao', 'temp_minBilbao', 'temp_maxBilbao', 'pressureBilbao', 'weather_idBilbao', 'tempMadrid', 'temp_minMadrid', 'temp_maxMadrid', 'temp_minSeville', 'pressureSeville', 'weather_idSeville', 'tempValencia', 'temp_minValencia', 'coal_oil_fossil_MWh', 'renewables_MWh', 'generation biomass_MWh', 'forecast solar day ahead_MWh', 'forecast wind onshore day ahead_MWh']

completeDataset = completeDataset[relevant_features]


# Using seaborn for heatmap correlation matrix

# Plot size
fig, ax = plt.subplots(figsize=(40,20))

# This method will only be used for continuous variables
continuousVariables = completeDataset.select_dtypes('float64','int64')
heatmap = sn.heatmap(completeDataset.corr(), annot=True, fmt='.2f')

plt.title('Heatmap for continuous variables', fontsize=20)
plt.savefig('Heatmap.png')
plt.show()
# Discarding features

completeDataset = completeDataset.drop(['LatitudeBilbao', 'LongitudeBilbao', 'LatitudeValencia', 'LongitudeValencia', 'LatitudeMadrid', 'LongitudeMadrid', 'Latitude Barcelona', 'Longitude Barcelona', 'LatitudeSeville', 'LongitudeSeville'], axis = 1)
# Rank 2D Pearson Algorithm

# Plot Size
plt.figure(figsize=(20, 15))


continuousVariables = completeDataset.select_dtypes('float64','int64')


# Definition of the algorithm
visualizer = Rank2D(algorithm='pearson')
visualizer.fit_transform(continuousVariables)
visualizer.poof()
plt.show()
# Definition of the covariance
visualizer = Rank2D(features=continuousVariables.columns, algorithm='covariance')

# Plot Size
plt.figure(figsize=(20, 15))


label = continuousVariables['price actual_€/Mwh']

visualizer.fit(continuousVariables, label) 
visualizer.transform(continuousVariables) 
visualizer.poof()

plt.show()
# Defining the algorithm
visualizer = Rank1D(features=continuousVariables.columns, algorithm='shapiro')
visualizer.fit(continuousVariables, label) 
# Transforming data
visualizer.transform(continuousVariables)
visualizer.poof()
plt.show()
# Getting all relevant features 

completeDataset = completeDataset.drop(['total load forecast_MWh', 'price day ahead_€/Mwh', 'forecast solar day ahead_MWh', 'forecast wind onshore day ahead_MWh', 'date', 'Hour_Minute'], axis = 1)
#Importing the model
model = lgb.LGBMRegressor(objective= 'regression')
print('LightGBM Parameters:', np.array(model.get_params))

# Define features and label

features = completeDataset.drop(['price actual_€/Mwh'], axis=1)
label = completeDataset['price actual_€/Mwh']


# Train and test split
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2)



# Fit with train set
model.fit(X_train, y_train)
# Predictions to test data
ExpectedValues  = y_test
PredictedValues = model.predict(X_test)

# R2 and Mean Square Error for LightGBM Regressor

print('R2 Score: \n', metrics.r2_score(ExpectedValues, PredictedValues))
print('Mean Square Error: \n', metrics.mean_squared_log_error(ExpectedValues, PredictedValues))

# Expected vs Predicted Plot in seaborn values regressor plot
sn.regplot(ExpectedValues, PredictedValues, fit_reg=True, scatter_kws={'s': 100})

plt.title ('Expected vs Predicted Values with LightGBM model')
plt.show()
# Parameter Tuning

# Nº of CV folds
numberFolds = 5

# Validation CV function
def validationcv(parameterTuning):
    folds = KFold(numberFolds, shuffle=True).get_n_splits(X_train.values)
    rsquare= np.sqrt(-cross_val_score(model, X_train.values, y_train, scoring='neg_mean_squared_error', cv=folds))
    return(rsquare)


# Choosing parameters
parameters = {
    'objective':'regression',
    'boosting_type':'gbdt', 
    'max_bin':50,
    'num_leaves':3,
    'max_depth':10,
    'learning_rate':0.5, 
    'bagging_fraction':0.7,
    'bagging_freq':6,
    'bagging_seed':7,
    'min_data_in_leaf':5, 
    'min_sum_hessian_in_leaf':7}

# Setting model with the chosen parameters
parameterTuning = lgb.LGBMRegressor(**parameters)

# Fitting with new parameters
parameterTuning.fit(X_train, y_train)


# Expected and Predicted Values
ExpectedValues =  y_test
PredictedValues = parameterTuning.predict(X_test)

# Results
print('New R2 Score: \n', metrics.r2_score(ExpectedValues, PredictedValues))
print('New Mean Square Error: \n', metrics.mean_squared_log_error(ExpectedValues, PredictedValues))
# Model Deploy
print('Saving the model...')
model.booster_.save_model('LightgbmEnergyPricePrediction_Project.txt')