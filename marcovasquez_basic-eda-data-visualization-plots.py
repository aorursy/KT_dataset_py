import pandas as pd



import numpy as np

import os

import seaborn as sns



import matplotlib.pyplot as plt



from statsmodels.tsa.stattools import adfuller
!ls ../input/ashrae-energy-prediction/
print('Total File sizes')

print('-'*10)

for f in os.listdir('../input/ashrae-energy-prediction'):

    if 'zip' not in f:

        print(f.ljust(30) + str(round(os.path.getsize('../input/ashrae-energy-prediction/' + f) / 10000000, 2)) + 'MB')
%%time

train = pd.read_csv('/kaggle/input/ashrae-energy-prediction/train.csv',index_col= 'timestamp', parse_dates=True)

test = pd.read_csv('/kaggle/input/ashrae-energy-prediction/test.csv',index_col= 'timestamp', parse_dates=True)

sample_sub = pd.read_csv('/kaggle/input/ashrae-energy-prediction/sample_submission.csv')

building_metadata = pd.read_csv('/kaggle/input/ashrae-energy-prediction/building_metadata.csv')

weather_train = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_train.csv',index_col= 'timestamp', parse_dates=True)

train.head()
print('Train # rows: ',train.shape[0])

print('Train # Columns: ',train.shape[1])
test.head()
building_metadata.head()
print('building_metadata # rows: ',building_metadata.shape[0])

print('building_metadata # Columns: ',building_metadata.shape[1])
weather_train.head()
print('weather_train # rows: ',weather_train.shape[0])

print('weather_train # Columns: ',weather_train.shape[1])
sample_sub.head()
sample_sub.shape
print(list(train.columns))
print(list(test.columns))
print(list(building_metadata.columns))
print(list(sample_sub.columns))
train.info()
test.info()
building_metadata.info()
nulls = building_metadata.isnull().sum() # Sum of missing values

nulls = nulls[nulls > 0]  

nulls.sort_values(inplace=True)

nulls
weather_train.info()
nullsWeather = weather_train.isnull().sum() # Sum of missing values

nullsWeather = nullsWeather[nullsWeather > 0]  

nullsWeather.sort_values(inplace=True)

nullsWeather
building_metadata.head()
plt.figure(figsize=(7,7))

 

sns.countplot(y= building_metadata.primary_use,palette="Set2")
plt.figure(figsize=(7,7))

 

sns.countplot(y= building_metadata.site_id,palette="Set2")


sns.distplot(building_metadata.year_built, bins=25, hist=True,kde=False, rug=False ).set_title("Histogram of Year Built")
sns.distplot(building_metadata.square_feet, bins=25, hist=True,kde=False, rug=False ).set_title("Histogram of square_feet")
sns.distplot(building_metadata.floor_count, bins=25, hist=True,kde=False, rug=False ).set_title("Histogram of floor_count")
building_metadata.building_id.unique()
sns.distplot(weather_train.air_temperature, bins=25, hist=True,kde=False, rug=False ).set_title("Histogram of Air Temperature")
sns.distplot(weather_train.sea_level_pressure, bins=25, hist=True,kde=False, rug=False ).set_title("Histogram of Sea Level Pressure")
sns.distplot(weather_train.wind_speed, bins=25, hist=True,kde=False, rug=False ).set_title("Histogram of wind Speed")
Weather = weather_train.copy()
Weather.head()
trainsite1 = train[train['building_id'] == 0]
trainsite1.meter_reading.plot(figsize=(16,8))
site1 = Weather[Weather['site_id'] == 0 ]
site1.air_temperature.plot(figsize=(16,8))


site1.air_temperature.plot(kind = 'kde')
def test_stationa(data):



    rolmean = data.rolling(window = 10).mean()

    

    #plotting rolling statistics

    original = plt.plot(data, color = 'blue', label = 'Original')

    mean = plt.plot(rolmean, color = 'red', label = 'Rolling Mean')

    

    plt.legend()

    plt.title('Rolling  Mean')

    plt.show()

    

    
test_stationa(site1.air_temperature)
site1.dew_temperature.plot(figsize=(16,8))
site1.sea_level_pressure.plot(figsize=(16,8))
plt.figure(figsize=(11,11))

correlations = Weather.corr()

mask = np.zeros_like(correlations)

mask[np.triu_indices_from(mask)] = True 

with sns.axes_style("white"):

    ax = sns.heatmap(correlations, mask=mask, vmax=.9, square=True)