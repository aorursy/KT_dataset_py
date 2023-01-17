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
import zipfile



zf = zipfile.ZipFile('../input/bosch-production-line-performance/train_date.csv.zip') 

train_date_chunks = pd.read_csv(zf.open('train_date.csv'), iterator=True, chunksize=100000)



pd.options.display.max_columns = None

pd.options.display.max_rows = None
def get_date_frame():

    for data_frame in train_date_chunks:

        yield data_frame

        

get_df_date = get_date_frame()
df_date = next(get_df_date)
df_date.info()
df_date.describe()
df_date.head()
station_list = []

first_features_in_each_station = [] 



df_date_columns = df_date.columns.tolist()



for feature in df_date_columns[1:]:

    station = feature[:feature.index('_D')]

    if station in station_list:

        continue

    else:

        station_list.append(station)

        first_features_in_each_station.append(feature)
station_df_time_stamp = pd.DataFrame (np.array(df_date[first_features_in_each_station]), columns = station_list)
print("Total number of visited stations: {} calculated from all given parts".format(len(station_df_time_stamp.columns)))
station_df_time_stamp['Station list'] = station_df_time_stamp.stack().reset_index(level=1).groupby(level=0, sort=False)['level_1'].apply(list)

station_df_time_stamp.insert(0, "Id", np.array(df_date[["Id"]]))

station_df_time_stamp.insert(1, "#_of_S",station_df_time_stamp.count(axis=1)-2)
station_df_time_stamp.head()
station_one_hot = pd.DataFrame (np.array(df_date[first_features_in_each_station]), columns = station_list)

station_one_hot = station_one_hot.notnull().astype('int')

station_one_hot.insert(0, "Id", np.array(df_date[["Id"]]))
station_one_hot.head()
station_df_time_stamp.to_csv('stations_date_train.csv', index=False)  

station_one_hot.to_csv('stations_one_hot_train.csv', index=False)
while True:

    try:

        df_date = next(get_df_date)

    except:

        break

    

   # station with timestamp

    station_df_time_stamp = pd.DataFrame (np.array(df_date[first_features_in_each_station]), columns = station_list)



    station_df_time_stamp['Station list'] = station_df_time_stamp.stack().reset_index(level=1).groupby(level=0, sort=False)['level_1'].apply(list)

    station_df_time_stamp.insert(0, "Id", np.array(df_date[["Id"]]))

    station_df_time_stamp.insert(1, "#_of_S",station_df_time_stamp.count(axis=1)-2)

    

    with open("./stations_date_train.csv", 'a') as f:

        station_df_time_stamp.to_csv(f, mode='a', header=False, index=False)

    

    

    # station one hot

    station_one_hot = pd.DataFrame (np.array(df_date[first_features_in_each_station]), columns = station_list)

    station_one_hot = station_one_hot.notnull().astype('int')

    station_one_hot.insert(0, "Id", np.array(df_date[["Id"]]))

    

    with open("./stations_one_hot_train.csv", 'a') as f:

        station_one_hot.to_csv(f, mode='a', header=False, index=False)
zf = zipfile.ZipFile('../input/bosch-production-line-performance/test_date.csv.zip') 

test_date_chunks = pd.read_csv(zf.open('test_date.csv'), iterator=True, chunksize=100000)
def get_date_frame():

    for data_frame in test_date_chunks:

        yield data_frame

        

get_df_date = get_date_frame()
df_date = next(get_df_date)
station_list = []

first_features_in_each_station = [] 



df_date_columns = df_date.columns.tolist()



for feature in df_date_columns[1:]:

    station = feature[:feature.index('_D')]

    if station in station_list:

        continue

    else:

        station_list.append(station)

        first_features_in_each_station.append(feature)
station_df_time_stamp = pd.DataFrame (np.array(df_date[first_features_in_each_station]), columns = station_list)
print("Total number of visited stations: {} calculated from all given parts".format(len(station_df_time_stamp.columns)))
station_df_time_stamp['Station list'] = station_df_time_stamp.stack().reset_index(level=1).groupby(level=0, sort=False)['level_1'].apply(list)

station_df_time_stamp.insert(0, "Id", np.array(df_date[["Id"]]))

station_df_time_stamp.insert(1, "#_of_S",station_df_time_stamp.count(axis=1)-2)
station_one_hot = pd.DataFrame (np.array(df_date[first_features_in_each_station]), columns = station_list)

station_one_hot = station_one_hot.notnull().astype('int')

station_one_hot.insert(0, "Id", np.array(df_date[["Id"]]))
station_df_time_stamp.to_csv('stations_date_test.csv', index=False)  

station_one_hot.to_csv('stations_one_hot_test.csv', index=False)
station_one_hot.head()
while True:

    try:

        df_date = next(get_df_date)

    except:

        break

    

   # station with timestamp

    station_df_time_stamp = pd.DataFrame (np.array(df_date[first_features_in_each_station]), columns = station_list)



    station_df_time_stamp['Station list'] = station_df_time_stamp.stack().reset_index(level=1).groupby(level=0, sort=False)['level_1'].apply(list)

    station_df_time_stamp.insert(0, "Id", np.array(df_date[["Id"]]))

    station_df_time_stamp.insert(1, "#_of_S",station_df_time_stamp.count(axis=1)-2)

    

    with open("./stations_date_test.csv", 'a') as f:

        station_df_time_stamp.to_csv(f, mode='a', header=False, index=False)

    

    

    # station one hot

    station_one_hot = pd.DataFrame (np.array(df_date[first_features_in_each_station]), columns = station_list)

    station_one_hot = station_one_hot.notnull().astype('int')

    station_one_hot.insert(0, "Id", np.array(df_date[["Id"]]))

    

    with open("./stations_one_hot_test.csv", 'a') as f:

        station_one_hot.to_csv(f, mode='a', header=False, index=False)