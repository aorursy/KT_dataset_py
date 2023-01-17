import numpy as np

import pandas as pd

import scipy.special

import matplotlib.pyplot as plt

import os
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder
path_in = '../input/ashrae-energy-prediction/'

print(os.listdir(path_in))
train_data = pd.read_csv(path_in+'train.csv', parse_dates=['timestamp'])

train_weather = pd.read_csv(path_in+'weather_train.csv', parse_dates=['timestamp'])

building_data = pd.read_csv(path_in+'building_metadata.csv')
def plot_bar(data, name):

    fig = plt.figure(figsize=(16, 9))

    ax = fig.add_subplot(111)

    data_label = data[name].value_counts()

    dict_train = dict(zip(data_label.keys(), ((data_label.sort_index())).tolist()))

    names = list(dict_train.keys())

    values = list(dict_train.values())

    plt.bar(names, values)

    ax.set_xticklabels(names, rotation=45)

    plt.grid()

    plt.show()
print('# samples train_data:', len(train_data))

print('# samples train_weather:', len(train_weather))

print('# samples building_data:', len(building_data))
train_data.head()
train_weather.head()
building_data.head()
cols_with_missing_train_data = [col for col in train_data.columns if train_data[col].isnull().any()]

cols_with_missing_train_weather = [col for col in train_weather.columns if train_weather[col].isnull().any()]

cols_with_missing_building = [col for col in building_data.columns if building_data[col].isnull().any()]
print(cols_with_missing_train_data)

print(cols_with_missing_train_weather)

print(cols_with_missing_building)
train_data['month'] = train_data['timestamp'].dt.month

train_data['day'] = train_data['timestamp'].dt.weekday

train_data['hour'] = train_data['timestamp'].dt.hour
train_data['weekend'] = np.where((train_data['day'] == 5) | (train_data['day'] == 6), 1, 0)
features_cyc = {'month' : 12, 'day' : 7, 'hour' : 24}

for feature in features_cyc.keys():

    train_data[feature+'_sin'] = np.sin((2*np.pi*train_data[feature])/features_cyc[feature])

    train_data[feature+'_cos'] = np.cos((2*np.pi*train_data[feature])/features_cyc[feature])

train_data = train_data.drop(features_cyc.keys(), axis=1)
train_data = pd.get_dummies(train_data, columns=['meter'])
train_data.head()
imp_most = SimpleImputer(strategy='most_frequent')
building_data[cols_with_missing_building] = imp_most.fit_transform(building_data[cols_with_missing_building])
plot_bar(building_data, 'primary_use')
map_use = dict(zip(building_data['primary_use'].value_counts().sort_index().keys(),

                     range(1, len(building_data['primary_use'].value_counts())+1)))
building_data['primary_use'] = building_data['primary_use'].replace(map_use)
building_data = pd.get_dummies(building_data, columns=['primary_use'])
building_scale = ['square_feet', 'year_built', 'floor_count']
mean = building_data[building_scale].mean(axis=0)

building_data[building_scale] = building_data[building_scale].astype('float32')

building_data[building_scale] -= building_data[building_scale].mean(axis=0)

std = building_data[building_scale].std(axis=0)

building_data[building_scale] /= building_data[building_scale].std(axis=0)
building_data.head()
weather_int = ['cloud_coverage']

weather_cyc = ['wind_direction']

weather_scale = ['air_temperature', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure', 'wind_speed']
imp_most = SimpleImputer(strategy='most_frequent')

train_weather[cols_with_missing_train_weather] = imp_most.fit_transform(train_weather[cols_with_missing_train_weather])
train_weather['wind_direction'+'_sin'] = np.sin((2*np.pi*train_weather['wind_direction'])/360)

train_weather['wind_direction'+'_cos'] = np.cos((2*np.pi*train_weather['wind_direction'])/360)

train_weather = train_weather.drop(['wind_direction'], axis=1)
mean = train_weather[weather_scale].mean(axis=0)

train_weather[weather_scale] = train_weather[weather_scale].astype('float32')

train_weather[weather_scale] -= train_weather[weather_scale].mean(axis=0)

std = train_weather[weather_scale].std(axis=0)

train_weather[weather_scale] /= train_weather[weather_scale].std(axis=0)
train_weather.head()
train_data = pd.merge(train_data, building_data, on='building_id', right_index=True)

train_data = train_data.sort_values(['timestamp'])

train_data = pd.merge_asof(train_data, train_weather, on='timestamp', by='site_id', right_index=True)

del building_data

del train_weather
train_data = train_data.sort_index()
no_feature = ['building_id', 'timestamp', 'meter_reading', 'site_id']
X_train = train_data[train_data.columns.difference(no_feature)].copy(deep=False)

y_train = train_data['meter_reading']
del train_data
X_train.head()
y_train_scaled = np.log1p(y_train)
y_train_scaled.hist(bins=50)
y_train_scaled[110:115]
y_test = np.expm1(y_train_scaled)
y_test[110:115]
def rmse(y_true, y_pred):

    """ root_mean_squared_error """

    return K.sqrt(K.mean(K.square(y_pred - y_true)))