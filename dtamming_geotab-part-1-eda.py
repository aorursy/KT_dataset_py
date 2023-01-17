import gc

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import scipy.stats as stats

import statsmodels.api as sm

from tqdm import tqdm

import time



from sklearn.metrics import mean_squared_error



from xgboost import XGBRegressor

from catboost import CatBoostRegressor



import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



np.random.seed(314)
other_columns = ['TimeFromFirstStop_p20', 'TimeFromFirstStop_p40', 'TimeFromFirstStop_p50', 'TimeFromFirstStop_p60', 'TimeFromFirstStop_p80',

                 'TotalTimeStopped_p40', 'TotalTimeStopped_p60', 'DistanceToFirstStop_p40', 'DistanceToFirstStop_p60']

train = pd.read_csv('../input/bigquery-geotab-intersection-congestion/train.csv').set_index('RowId').drop(columns=other_columns)

test = pd.read_csv('../input/bigquery-geotab-intersection-congestion/test.csv').set_index('RowId')

train_idxs = train.index

test_idxs = test.index

data = pd.concat([train, test], axis=0, join='outer')
target_vars = ['TotalTimeStopped_p20', 'TotalTimeStopped_p50', 'TotalTimeStopped_p80', 'DistanceToFirstStop_p20', 'DistanceToFirstStop_p50', 'DistanceToFirstStop_p80']

cities = train.City.unique()

fig, ax = plt.subplots(nrows=6, ncols=4, figsize=(20,30))

bins = list(range(0, 200, 10))

for i, var in enumerate(target_vars):

    for j, city in enumerate(cities):

        sns.distplot(train[train.City == city][var], bins=bins, kde=False, ax=ax[i, j]).set_title(city)
data.isna().sum().sort_values() * 100 / len(data)
roadways = ['Street', 'Road', 'Boulevard', 'Avenue', 'Lane', 'Drive', 'Parkway', 'Place', 'Way', 

            'Circle', 'Highway', 'Pkwy', 'St', 'Connector', 'Broadway', 'Overpass', 'Ave', 'Square', 

            'Tunner', 'Rd', 'Bld', 'Bridge', 'Expressway', 'Pike']

to_longform = {'Rd': 'Road', 'Bld': 'Boulevard', 'Ave': 'Avenue', 'St': 'Street', 'Pkwy': 'Parkway'}



street_names = pd.concat([data['EntryStreetName'], data['ExitStreetName']], ignore_index=True).dropna()



seen = set()

for street in street_names:

    if all([roadway not in street for roadway in roadways]):

        if street not in seen:

            print(street)

            seen.add(street)
def to_roadway(StreetName):

    if pd.isnull(StreetName):

        return 'Other'

    for roadway in roadways:

        if roadway in StreetName:

            if roadway in to_longform:

                return to_longform[roadway]

            else:

                return roadway

    return 'Other'



both_roadway = street_names.apply(to_roadway)
plt.figure(figsize=(20,10))

sns.countplot(both_roadway, order=both_roadway.value_counts().index);
roadways = ['Street', 'Avenue', 'Road', 'Boulevard', 'Drive', 'Parkway']

to_longform = {'St': 'Street', 'Ave': 'Avenue', 'Rd': 'Road', 'Bld': 'Boulevard', 'Pkwy': 'Parkway'}

def to_roadway(StreetName):

    if pd.isnull(StreetName):

        return 'Other'

    for roadway in roadways:

        if roadway in StreetName:

            if roadway in to_longform:

                return to_longform[roadway]

            else:

                return roadway

    return 'Other'
data['EntryRoadway'] = data['EntryStreetName'].apply(to_roadway)

data['ExitRoadway'] = data['ExitStreetName'].apply(to_roadway)

data.drop(columns=['EntryStreetName', 'ExitStreetName'], inplace=True)
fig, ax = plt.subplots(ncols=2, figsize=(20, 6))

sns.countplot(data.EntryRoadway, order=data.EntryRoadway.value_counts().index, ax=ax[0]);

sns.countplot(data.ExitRoadway, order=data.ExitRoadway.value_counts().index, ax=ax[1]);
radians_map = dict(zip('E NE N NW W SW S SE'.split(), [np.pi*i/4 for i in range(8)]))

degrees_map = dict(zip('E NE N NW W SW S SE'.split(), [i*45 for i in range(8)]))

entry_heading_degrees = data.EntryHeading.map(degrees_map)

exit_heading_degrees = data.ExitHeading.map(degrees_map)
fig, ax = plt.subplots(ncols=2, figsize=(22,6))

data['DeltaHeading'] = (exit_heading_degrees - entry_heading_degrees + 180) % 360 - 180

sns.countplot(x='DeltaHeading', data=data, order=sorted(data.DeltaHeading.unique()), ax=ax[0]);

data['TotalTimeStopped_p80_log'] = np.log1p(data['TotalTimeStopped_p80'])

sns.boxplot(x='DeltaHeading', y='TotalTimeStopped_p80_log', data=data, ax=ax[1]);
pooled_heading_map = {-180: 'L', -135:'R', -90:'R', -45:'R_soft', 0:'S', 45:'L_soft', 90:'L', 135:'L'}

data['DeltaHeading'] = data['DeltaHeading'].map(pooled_heading_map)

data.drop(columns=['ExitHeading'], inplace=True)
fig, ax = plt.subplots(ncols=2, figsize=(22,6))

sns.countplot(x='Month', data=data, ax=ax[0]);

sns.violinplot(x='Month', y='TotalTimeStopped_p80_log', data=data, ax=ax[1]);
fig, ax = plt.subplots(ncols=2, figsize=(22,6))

sns.countplot(x='Hour', data=data, ax=ax[0]);

sns.boxplot(x='Hour', y='TotalTimeStopped_p80_log', data=data, ax=ax[1]);
data['dist_to_5pm'] = abs(data.Hour - 17)

data['dist_to_8am'] = abs(data.Hour - 8)
fig, ax = plt.subplots(ncols=2, nrows=4, figsize=(20, 20))

for row, city in enumerate(data.City.unique()):

    city_data = data[(data.City == city) & (data.TotalTimeStopped_p80_log > 0)]

    sns.scatterplot(x='Longitude', y='TotalTimeStopped_p80', data=city_data, ax=ax[row, 0]).set_title(city);

    sns.scatterplot(x='Latitude', y='TotalTimeStopped_p80', data=city_data, ax=ax[row, 1]).set_title(city);
centers_data = [['Atlanta', 33.7490, -84.3880], ['Boston', 42.3601, -71.0589], ['Chicago', 41.8781, -87.6298], ['Philadelphia', 39.9509, -75.1575]]

centers = pd.DataFrame(centers_data, columns=['City', 'Latitude', 'Longitude']).set_index('City')



data['latitude_dist'] = data[['City', 'Latitude']].apply(lambda x : abs(x['Latitude'] - centers.loc[x['City'], 'Latitude']), axis=1)

data['longitude_dist'] = data[['City', 'Longitude']].apply(lambda x : abs(x['Longitude'] - centers.loc[x['City'], 'Longitude']), axis=1)
data.drop(columns=['Latitude', 'Longitude'], inplace=True)
data.drop(columns=['Path'], inplace=True)
data['city_intersection'] = data.City + data.IntersectionId.astype(str)

data.drop(columns=['IntersectionId'], inplace=True)
data.drop(columns=['TotalTimeStopped_p80_log'], inplace=True)
num_vars = ['dist_to_5pm', 'dist_to_8am', 'latitude_dist', 'longitude_dist']

cat_vars = ['EntryRoadway', 'ExitRoadway', 'DeltaHeading', 'EntryHeading', 'Hour', 'Month', 'city_intersection']

bool_vars = ['Weekend']

vars_type_map = {'num': num_vars, 'cat': cat_vars, 'bool': bool_vars}

predictor_vars = [var for L in vars_type_map.values() for var in L]

target_vars = ['TotalTimeStopped_p20', 'TotalTimeStopped_p50', 'TotalTimeStopped_p80', 'DistanceToFirstStop_p20', 'DistanceToFirstStop_p50', 'DistanceToFirstStop_p80']
data[vars_type_map['cat']] = data[vars_type_map['cat']].astype('category')

data.loc[train_idxs].to_csv('train_processed.csv')

data.loc[test_idxs].to_csv('test_processed.csv')