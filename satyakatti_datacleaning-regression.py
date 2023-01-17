import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import normalize
from sklearn.preprocessing import Normalizer
from scipy.stats import boxcox

import matplotlib.pyplot as plt

%matplotlib inline
weather_data = pd.read_csv('../input/Summary of Weather.csv')
#Lets first see a sample of the data
weather_data.sample(5)
#Let us understand the nature of data and distribution of it.
weather_data.describe()

#Ok not such anomalies to naked eyes
#Let us see how categorical data is
weather_data.describe(include=['O'])
#There are 119040 rows and 31 columns.
print (weather_data.shape)
print ('--'*30)

#Lets see what percentage of each column has null values
#It means count number of nulls in every column and divide by total num of rows.

print (weather_data.isnull().sum()/weather_data.shape[0] * 100)
#weather_data[col].isnull().sum()/weather_data.shape[0] * 100 < 70)
#Commented line to check if that column's null percentage is below 70

cols = [col for col in weather_data.columns if (weather_data[col].isnull().sum()/weather_data.shape[0] * 100 < 70)]
weather_data_trimmed = weather_data[cols]

#STA is more of station code, lets drop it for the moment
weather_data_trimmed = weather_data_trimmed.drop(['STA'], axis=1)

print ('Legitimate columns after dropping null columns: %s' % weather_data_trimmed.shape[1])
weather_data_trimmed.isnull().sum()
weather_data_trimmed.sample(5)
#Check dtypes and look for conversion if needed
weather_data_trimmed.dtypes

#Looks like some columns needs to be converted to numeric field

weather_data_trimmed['Snowfall'] = pd.to_numeric(weather_data_trimmed['Snowfall'], errors='coerce')
weather_data_trimmed['SNF'] = pd.to_numeric(weather_data_trimmed['SNF'], errors='coerce')
weather_data_trimmed['PRCP'] = pd.to_numeric(weather_data_trimmed['PRCP'], errors='coerce')
weather_data_trimmed['Precip'] = pd.to_numeric(weather_data_trimmed['Precip'], errors='coerce')

weather_data_trimmed['Date'] = pd.to_datetime(weather_data_trimmed['Date'])
#Fill remaining null values. FOr the moment lts perform ffill

weather_data_trimmed.fillna(method='ffill', inplace=True)
weather_data_trimmed.fillna(method='bfill', inplace=True)

weather_data_trimmed.isnull().sum()
#Well no more NaN and null values to worry about
print (weather_data_trimmed.dtypes)
print ('--'*30)
weather_data_trimmed.sample(3)
#weather_data_trimmed_scaled = minmax_scale(weather_data_trimmed.iloc[:, 1:])

weather_data_trimmed['Precip_scaled'] = minmax_scale(weather_data_trimmed['Precip'])
weather_data_trimmed['MeanTemp_scaled'] = minmax_scale(weather_data_trimmed['MeanTemp'])
weather_data_trimmed['YR_scaled'] = minmax_scale(weather_data_trimmed['YR'])
weather_data_trimmed['Snowfall_scaled'] = minmax_scale(weather_data_trimmed['Snowfall'])
weather_data_trimmed['MAX_scaled'] = minmax_scale(weather_data_trimmed['MAX'])
weather_data_trimmed['MIN_scaled'] = minmax_scale(weather_data_trimmed['MIN'])

#weather_data_trimmed.sample(3)
#Plot couple of columns to see how the data is scaled

fig, ax = plt.subplots(4, 2, figsize=(15, 15))

sns.distplot(weather_data_trimmed['Precip'], ax=ax[0][0])
sns.distplot(weather_data_trimmed['Precip_scaled'], ax=ax[0][1])

sns.distplot(weather_data_trimmed['MeanTemp'], ax=ax[1][0])
sns.distplot(weather_data_trimmed['MeanTemp_scaled'], ax=ax[1][1])

sns.distplot(weather_data_trimmed['Snowfall'], ax=ax[2][0])
sns.distplot(weather_data_trimmed['Snowfall_scaled'], ax=ax[2][1])

sns.distplot(weather_data_trimmed['MAX'], ax=ax[3][0])
sns.distplot(weather_data_trimmed['MAX_scaled'], ax=ax[3][1])

Precip_norm = boxcox(weather_data_trimmed['Precip_scaled'].loc[weather_data_trimmed['Precip_scaled'] > 0])
MeanTemp_norm = boxcox(weather_data_trimmed['MeanTemp_scaled'].loc[weather_data_trimmed['MeanTemp_scaled'] > 0])
YR_norm = boxcox(weather_data_trimmed['YR_scaled'].loc[weather_data_trimmed['YR_scaled'] > 0])
Snowfall_norm = boxcox(weather_data_trimmed['Snowfall_scaled'].loc[weather_data_trimmed['Snowfall_scaled'] > 0])
MAX_norm = boxcox(weather_data_trimmed['MAX_scaled'].loc[weather_data_trimmed['MAX_scaled'] > 0])
MIN_norm = boxcox(weather_data_trimmed['MIN_scaled'].loc[weather_data_trimmed['MIN_scaled'] > 0])
fig, ax = plt.subplots(4, 2, figsize=(15, 15))

sns.distplot(weather_data_trimmed['Precip_scaled'], ax=ax[0][0])
sns.distplot(Precip_norm[0], ax=ax[0][1])

sns.distplot(weather_data_trimmed['MeanTemp_scaled'], ax=ax[1][0])
sns.distplot(MeanTemp_norm[0], ax=ax[1][1])

sns.distplot(weather_data_trimmed['Snowfall_scaled'], ax=ax[2][0])
sns.distplot(Snowfall_norm[0], ax=ax[2][1])

sns.distplot(weather_data_trimmed['MAX_scaled'], ax=ax[3][0])
sns.distplot(MAX_norm[0], ax=ax[3][1])

