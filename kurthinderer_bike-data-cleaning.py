import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

import os

import geopandas as gpd

import json

import datetime



%matplotlib inline
path = '/kaggle/input/bikedata/Bike/'



filenames = os.listdir(path)

bike_load_dfs = []

for f in filenames:

    bike_load_dfs.append(pd.read_csv(path + f))



len(bike_load_dfs)
for df in bike_load_dfs:

    print(df.shape)

    print(df.columns)
bike_df = bike_load_dfs[0].copy()

for i in range(1, len(bike_load_dfs)):

    bike_df = bike_df.append(bike_load_dfs[i])



bike_df.shape
bike_df.head()
bike_df.dtypes
bike_df = bike_df.drop(labels = ['rental_access_method', 'bike_share_for_all_trip'], axis = 1)

bike_df.shape
bike_df.to_csv('bike_uncleaned.csv', index = False)
bike_df.info()
for col in bike_df.columns:

    print(col + ' non-null: ' + str(bike_df[col].count()))
bike_df.describe()
bike_df.nunique()
#looking at the durrations via a log 

duration_log = bike_df['duration_sec'].apply(np.log10)

duration_log.head()
duration_log.hist()
print('Number 1 second or more: ' + str(bike_df[bike_df['duration_sec'] > 0].shape[0]))

print('Number 1 minute or more: ' + str(bike_df[bike_df['duration_sec'] > 60].shape[0]))

print('Number 1 hour or more: ' + str(bike_df[bike_df['duration_sec'] > 3600].shape[0]))

print('Number 1 day or more: ' + str(bike_df[bike_df['duration_sec'] > 86400].shape[0]))

print('Number 1 week or more: ' + str(bike_df[bike_df['duration_sec'] > 604800].shape[0]))
# look at the zero longitude and latitude

bike_df[bike_df['start_station_longitude'] == 0].shape[0]
bike_df[bike_df['start_station_longitude'] == 0]['start_station_id'].value_counts()
bike_df[bike_df['start_station_latitude'] == 0].shape[0]
bike_df[bike_df['start_station_latitude'] == 0]['start_station_longitude'].value_counts()
bike_df[bike_df['end_station_longitude'] == 0].shape[0]
bike_df[bike_df['end_station_longitude'] == 0]['end_station_id'].value_counts()
bike_df[bike_df['end_station_latitude'] == 0].shape[0]
bike_df[bike_df['end_station_latitude'] == 0]['end_station_longitude'].value_counts()
bike_df[bike_df['start_station_id'] == 420].shape[0]
bike_df[bike_df['start_station_id'] == 449].shape[0]
bike_df[bike_df['end_station_id'] == 420].shape[0]
bike_df[bike_df['end_station_id'] == 449].shape[0]
bike_df[(bike_df['start_station_id'] == 420) | 

        (bike_df['start_station_id'] == 449)][['start_station_id', 'start_station_name']].value_counts()
# Looking into duplicate station names.

stations = bike_df[['start_station_id', 'start_station_name']].drop_duplicates()

stations.shape
station_name_count = stations[['start_station_id', 'start_station_name']].groupby('start_station_id').agg('count')

station_name_count.shape
station_name_dup = station_name_count[station_name_count['start_station_name'] > 1]

station_name_dup.shape
bike_df[bike_df['start_station_id'].isin(station_name_dup.index)][['start_station_id','start_station_name']].value_counts()
for id in station_name_dup.index:

    print(bike_df[bike_df['start_station_id'] == id][['start_station_id','start_station_name']].value_counts())
stations_end = bike_df[['end_station_id', 'end_station_name']].drop_duplicates()

station_name_count_end = stations_end[['end_station_id', 'end_station_name']].groupby('end_station_id').agg('count')

station_name_dup_end = station_name_count_end[station_name_count_end['end_station_name'] > 1]

station_name_dup.index == station_name_dup_end.index
start_min_max = bike_df[['start_station_id', 'start_station_latitude', 

                         'start_station_longitude']].groupby('start_station_id').agg(['min', 'max'])

start_min_max.columns
start_min_max['lat_dif'] = start_min_max[('start_station_latitude', 'max')] - start_min_max[('start_station_latitude', 'min')]

start_min_max['lon_dif'] = start_min_max[('start_station_longitude', 'max')] - start_min_max[('start_station_longitude', 'min')]

start_min_max.head()
start_min_max['lat_dif'].max(), start_min_max['lon_dif'].max()
end_min_max = bike_df[['end_station_id', 'end_station_latitude', 

                         'end_station_longitude']].groupby('end_station_id').agg(['min', 'max'])

end_min_max['lat_dif'] = end_min_max[('end_station_latitude', 'max')] - end_min_max[('end_station_latitude', 'min')]

end_min_max['lon_dif'] = end_min_max[('end_station_longitude', 'max')] - end_min_max[('end_station_longitude', 'min')]

end_min_max['lat_dif'].max(), end_min_max['lon_dif'].max()
start_wrong_lat = start_min_max[start_min_max['lat_dif'] > 0.001].index

start_wrong_lon = start_min_max[start_min_max['lon_dif'] > 0.001].index

end_wrong_lat = end_min_max[end_min_max['lat_dif'] > 0.001].index

end_wrong_lon = end_min_max[end_min_max['lon_dif'] > 0.001].index



len(start_wrong_lat), len(start_wrong_lon), len(end_wrong_lat), len(end_wrong_lon)
start_wrong_lat, start_wrong_lon, end_wrong_lat, end_wrong_lon
for id in start_wrong_lon:

    print(bike_df[bike_df['start_station_id'] == id][['start_station_id','start_station_latitude', 

                                                     'start_station_longitude']].value_counts())
for id in end_wrong_lon:

    print(bike_df[bike_df['end_station_id'] == id][['end_station_id','end_station_latitude', 

                                                     'end_station_longitude']].value_counts())
#Let's make sure that the non-station locations are not out of state or on Mars.

bike_df[bike_df['start_station_id'].isnull()][['start_station_latitude', 'start_station_longitude']].describe()
bike_df[bike_df['end_station_id'].isnull()][['end_station_latitude', 'end_station_longitude']].describe()
bike_df[bike_df['start_station_latitude'] < 37].shape[0] + bike_df[bike_df['end_station_latitude'] < 37].shape[0]
bike_df[bike_df['start_station_latitude'] > 38].shape[0] + bike_df[bike_df['end_station_latitude'] > 38].shape[0]
bike_df[bike_df['start_station_longitude'] > -121].shape[0] + bike_df[bike_df['end_station_longitude'] > -121].shape[0]
no_name_start = bike_df[(bike_df['start_station_id'].isnull()) & 

                        (bike_df['start_station_name'].notnull())][['start_time', 'start_station_name']]

#added start time so I have something to count on.

no_name_start.head()
no_name_start.groupby('start_station_name').agg('count')
no_name_end = bike_df[(bike_df['end_station_id'].isnull()) & 

                        (bike_df['end_station_name'].notnull())][['start_time', 'end_station_name']]

no_name_end.groupby('end_station_name').agg('count')
for name in pd.unique(no_name_end['end_station_name']):

    print(name +"    "+ str(bike_df[(bike_df['end_station_id'].notnull()) & 

            (bike_df['end_station_name'] == name)]['end_station_name'].count()))
corrected_station = ['Green St at Van Ness Ave', 'Clement St at 32nd Ave']

for name in corrected_station:

    print(name +"    "+ str(bike_df[(bike_df['end_station_id'].notnull()) & 

            (bike_df['end_station_name'] == name)]['end_station_name'].count()))
bike_df_not_clean = bike_df.copy()
bike_df = bike_df_not_clean.copy()

bike_df.shape
#oops, there are multiple indexs that are the same. When I tried to drop by index I lost more rows then

#I should have. Let's reset that index first.

bike_df.set_index(np.arange(bike_df.shape[0]), drop = True, inplace = True)

bike_df.tail()
bike_df['start_time'] = pd.to_datetime(bike_df['start_time'])

bike_df['end_time'] = pd.to_datetime(bike_df['end_time'])

bike_df.dtypes
start_date = datetime.datetime(2019, 2, 11, 0, 0, 0)

end_date = datetime.datetime(2020, 3, 16, 0, 0, 0)



start_date, end_date
too_early_index = bike_df[bike_df['start_time'] < start_date].index

len(too_early_index)
bike_df.drop(index = too_early_index, inplace = True)

bike_df.shape
too_late_index = bike_df[bike_df['end_time'] > end_date].index

len(too_late_index)
bike_df.drop(index = too_late_index, inplace = True)

bike_df.shape
no_id = bike_df[(bike_df['end_station_id'].isnull()) & 

                        (bike_df['end_station_name'].notnull())][['end_station_name']].drop_duplicates()[

    'end_station_name']

no_id
bike_df[bike_df['end_station_name'].isin(no_id)][['end_station_id', 'end_station_name']].value_counts()
bike_df[bike_df['end_station_name'].isin(['Green St at Van Ness Ave',

                                         'Clement St at 32nd Ave'])][['end_station_id', 

                                                                      'end_station_name']].value_counts()
no_id_list = no_id.tolist()

no_id_list
#manually making a list of ids so I can match up the values and make a dictionary

correct_id = [496, 47, 323, 516, 289, 290, 16, 277, 316, 290, 378, 289, 321, 37, 234]

id_correction = {no_id_list[i]: correct_id[i] for i in range(len(no_id_list))}

id_correction
for key, value in id_correction.items():

    bike_df.loc[bike_df['start_station_name'] == key, 'start_station_id'] = value

    bike_df.loc[bike_df['end_station_name'] == key, 'end_station_id'] = value
for key in id_correction.items():

    print(bike_df[(bike_df['start_station_name'] == key) & 

            (bike_df['start_station_id']).isnull()][['start_station_name']].count())

    print(bike_df[(bike_df['end_station_name'] == key) & 

            (bike_df['end_station_id']).isnull()][['end_station_name']].count())
for col in ['start_station_id', 'start_station_name', 'end_station_id', 'end_station_name']:

    print(col + ': ' + str(bike_df[col].count()))
bike_df.fillna({'start_station_id': -1, 'start_station_name': 'non-station', 

                'end_station_id': -1, 'end_station_name': 'non-station'}, inplace = True)



for col in ['start_station_id', 'start_station_name', 'end_station_id', 'end_station_name']:

    print(col + ': ' + str(bike_df[col].count()))
zero_coord = bike_df[(bike_df['start_station_latitude'] == 0) | 

                     (bike_df['end_station_latitude'] == 0)].index

len(zero_coord)
#one less than I counted earlier, so let's see if that one got purged when I set the date range.

len(bike_df_not_clean[(bike_df_not_clean['start_station_latitude'] == 0) | 

                     (bike_df_not_clean['end_station_latitude'] == 0)].index)
#yeap, that's why I missed one. Now time to remove these from the bike dataframe.

bike_df.drop(index = zero_coord, inplace = True)

bike_df.shape
bike_df = bike_df.astype({'bike_id':'int64','start_station_id':'int64','end_station_id':'int64'})

bike_df.dtypes
id_name = {16: 'Market St at Steuart St', 37: 'Folsom St at 2nd St', 47: 'Clara St at 4th St', 

           224: '21st St at 5th Ave', 229: 'Bond St at High St', 234: 'Fruitvale Ave at International Blvd', 

           277: 'W Julian St at N Morrison St', 289: '5th St at Taylor St', 290: 'George St at 1st St', 

           316: '1st St at San Carlos St', 321: 'Folsom St at 5th St', 323: 'Broadway at Kearny St', 

           349: 'Howard St at 6th St', 378: '7th St at Empire St', 393: 'Asbury St at The Alameda', 

           516: 'Clement St at 32nd Ave', 496: 'Green St at Van Ness Ave'}



for key, value in id_name.items():

    bike_df.loc[bike_df['start_station_id'] == key, 'start_station_name'] = value

    bike_df.loc[bike_df['end_station_id'] == key, 'end_station_name'] = value



for key in id_name:

    print(bike_df[bike_df['start_station_id'] == key]['start_station_name'].value_counts())

    print(bike_df[bike_df['end_station_id'] == key]['end_station_name'].value_counts())
bike_df[bike_df['end_station_id'] == 408][['end_station_id','end_station_latitude', 

                                                     'end_station_longitude']].value_counts()
print(bike_df[bike_df['start_station_latitude'] == 45.510000].head())

print(bike_df[bike_df['end_station_latitude'] == 45.510000].head())
mont_index = bike_df[bike_df['start_station_latitude'] == 45.510000].index

bike_df.drop(index = mont_index, inplace = True)

bike_df.shape
stations_start = bike_df[bike_df['start_station_id'] != -1][['start_station_id', 

                                                             'start_station_latitude', 'start_station_longitude']]

stations_end = bike_df[bike_df['end_station_id'] != -1][['end_station_id', 

                                                             'end_station_latitude', 'end_station_longitude']]

stations_start.rename(columns = {'start_station_id':'station_id', 'start_station_latitude': 'station_latitude', 

                               'start_station_longitude':'station_longitude'}, inplace = True)

stations_start.columns
stations_end.rename(columns = {'end_station_id':'station_id', 'end_station_latitude': 'station_latitude', 

                               'end_station_longitude':'station_longitude'}, inplace = True)

stations = stations_start.append(stations_end)

stations.head()
stations_mean = stations.groupby('station_id').agg('mean')

stations_mean.head()
stations_mean.columns
lat_dict = {}

lon_dict = {}

for index in stations_mean.index:

    lat_dict.update({index: round(stations_mean['station_latitude'][index], 6)})

    lon_dict.update({index: round(stations_mean['station_longitude'][index], 6)})



lat_dict
for key in lat_dict:

    bike_df.loc[bike_df['start_station_id'] == key, 'start_station_latitude'] = lat_dict[key]

    bike_df.loc[bike_df['start_station_id'] == key, 'start_station_longitude'] = lon_dict[key]

    bike_df.loc[bike_df['end_station_id'] == key, 'end_station_latitude'] = lat_dict[key]

    bike_df.loc[bike_df['end_station_id'] == key, 'end_station_longitude'] = lon_dict[key]



bike_df[['start_station_id', 'start_station_latitude']].value_counts()
bike_df.head()
week_long_index = bike_df[bike_df['duration_sec'] > 604800].index

len(week_long_index)
bike_df.drop(index = week_long_index, inplace = True)

bike_df.shape
south = bike_df[(bike_df['start_station_latitude'] < 37) | (bike_df['end_station_latitude'] < 37)].index.tolist()

north = bike_df[(bike_df['start_station_latitude'] > 38) | (bike_df['end_station_latitude'] > 38)].index.tolist()

east = bike_df[(bike_df['start_station_longitude'] > -121) | (bike_df['end_station_longitude'] > -121)].index.tolist()



out_of_bounds = south + north + east

len(out_of_bounds)
bike_df.drop(index = out_of_bounds, inplace = True)

bike_df.shape
out_of_bounds
bike_df['start_time'].dt.date
stations_start = bike_df[bike_df['start_station_id'] != -1][['start_station_id', 'start_station_name', 

                                                             'start_station_latitude', 'start_station_longitude']]

stations_end = bike_df[bike_df['end_station_id'] != -1][['end_station_id', 'end_station_name', 

                                                             'end_station_latitude', 'end_station_longitude']]

stations_start.rename(columns = {'start_station_id':'station_id', 'start_station_name':'station_name', 

                                 'start_station_latitude': 'station_latitude', 

                                 'start_station_longitude':'station_longitude'}, inplace = True)

stations_end.rename(columns = {'end_station_id':'station_id', 'end_station_name':'station_name', 

                               'end_station_latitude': 'station_latitude', 

                               'end_station_longitude':'station_longitude'}, inplace = True)

stations_df = stations_start.append(stations_end)

stations_df.head()
stations_df.nunique()
stations_df.drop_duplicates(inplace = True)

stations_df.shape
bike_df.to_csv('bike.csv', index = False)

stations_df.to_csv('stations.csv', index = False)