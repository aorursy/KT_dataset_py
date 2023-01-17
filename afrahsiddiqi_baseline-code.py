# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
building_metadata = pd.read_csv('/kaggle/input/ashrae-energy-prediction/building_metadata.csv')

train = pd.read_csv('/kaggle/input/ashrae-energy-prediction/train.csv')

test = pd.read_csv('/kaggle/input/ashrae-energy-prediction/test.csv')

weather_train = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_train.csv')

weather_test = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_test.csv')
print('Size of the building_metadata dataset is', building_metadata.shape)

print('Size of the weather_train dataset is', weather_train.shape)

print('Size of the train dataset is', train.shape)

print('Size of the weather_test dataset is', weather_test.shape)

print('Size of the test dataset is', test.shape)
train.head()
print('The start date is', train.timestamp[0])

print('The last date is', train.timestamp[len(train) -1])
train.describe()
train.info()
print('The start date is', test.timestamp[0])

print('The last date is', test.timestamp[len(test) -1])
test.info()
building_metadata.head()
building_metadata.info()
# Also, each of the 1449 buildings has a site_id. We can determine the actual number of sites:

print(building_metadata.site_id.unique())

print(building_metadata.site_id.value_counts().sort_index())
weather_train.head()
print('The start date is', weather_train.timestamp[0])

print('The last date is', weather_train.timestamp[len(weather_train) -1])
weather_train.info()
print('The start date is', weather_test.timestamp[0])

print('The last date is', weather_test.timestamp[len(weather_test) -1])
weather_test.info()
# Elminiate missing values from weather datasets by replacing NaN values with mean values taken for a month. 

# Source code taken from, https://www.kaggle.com/aitude/ashrae-missing-weather-data-handling#How-to-Use



def missing_statistics(df):    

    statitics = pd.DataFrame(df.isnull().sum()).reset_index()

    statitics.columns=['COLUMN NAME',"MISSING VALUES"]

    statitics['TOTAL ROWS'] = df.shape[0]

    statitics['% MISSING'] = round((statitics['MISSING VALUES']/statitics['TOTAL ROWS'])*100,2)

    return statitics

    

def fill_weather_dataset(weather_df):

    

    # Find Missing Dates

    time_format = "%Y-%m-%d %H:%M:%S"

    start_date = datetime.datetime.strptime(weather_df['timestamp'].min(),time_format)

    end_date = datetime.datetime.strptime(weather_df['timestamp'].max(),time_format)

    total_hours = int(((end_date - start_date).total_seconds() + 3600) / 3600)

    hours_list = [(end_date - datetime.timedelta(hours=x)).strftime(time_format) for x in range(total_hours)]



    missing_hours = []

    for site_id in range(16):

        site_hours = np.array(weather_df[weather_df['site_id'] == site_id]['timestamp'])

        new_rows = pd.DataFrame(np.setdiff1d(hours_list,site_hours),columns=['timestamp'])

        new_rows['site_id'] = site_id

        weather_df = pd.concat([weather_df,new_rows])



        weather_df = weather_df.reset_index(drop=True)           



    # Add new Features

    weather_df["datetime"] = pd.to_datetime(weather_df["timestamp"])

    weather_df["day"] = weather_df["datetime"].dt.day

    weather_df["week"] = weather_df["datetime"].dt.week

    weather_df["month"] = weather_df["datetime"].dt.month

    

    # Reset Index for Fast Update

    weather_df = weather_df.set_index(['site_id','day','month'])



    air_temperature_filler = pd.DataFrame(weather_df.groupby(['site_id','day','month'])['air_temperature'].mean(),columns=["air_temperature"])

    weather_df.update(air_temperature_filler,overwrite=False)



    # Step 1

    cloud_coverage_filler = weather_df.groupby(['site_id','day','month'])['cloud_coverage'].mean()

    # Step 2

    cloud_coverage_filler = pd.DataFrame(cloud_coverage_filler.fillna(method='ffill'),columns=["cloud_coverage"])



    weather_df.update(cloud_coverage_filler,overwrite=False)



    due_temperature_filler = pd.DataFrame(weather_df.groupby(['site_id','day','month'])['dew_temperature'].mean(),columns=["dew_temperature"])

    weather_df.update(due_temperature_filler,overwrite=False)



    # Step 1

    sea_level_filler = weather_df.groupby(['site_id','day','month'])['sea_level_pressure'].mean()

    # Step 2

    sea_level_filler = pd.DataFrame(sea_level_filler.fillna(method='ffill'),columns=['sea_level_pressure'])



    weather_df.update(sea_level_filler,overwrite=False)



    wind_direction_filler =  pd.DataFrame(weather_df.groupby(['site_id','day','month'])['wind_direction'].mean(),columns=['wind_direction'])

    weather_df.update(wind_direction_filler,overwrite=False)



    wind_speed_filler =  pd.DataFrame(weather_df.groupby(['site_id','day','month'])['wind_speed'].mean(),columns=['wind_speed'])

    weather_df.update(wind_speed_filler,overwrite=False)



    # Step 1

    precip_depth_filler = weather_df.groupby(['site_id','day','month'])['precip_depth_1_hr'].mean()

    # Step 2

    precip_depth_filler = pd.DataFrame(precip_depth_filler.fillna(method='ffill'),columns=['precip_depth_1_hr'])



    weather_df.update(precip_depth_filler,overwrite=False)



    weather_df = weather_df.reset_index()

    weather_df = weather_df.drop(['datetime','day','week','month'],axis=1)

        

    return weather_df
missing_statistics(weather_train)
missing_statistics(weather_test)
weather_train = fill_weather_dataset(weather_train)

weather_test = fill_weather_dataset(weather_test)
# Missing Data from building_metadata

missing_statistics(building_metadata)
# floor count is missing too many values, probably best to delete it all together



del building_metadata['floor_count']

del building_metadata['year_built']
# Since the dataset sizes are quite large, it is important to reduce memory usage before proceeding

# Source code taken from https://www.kaggle.com/cereniyim/save-the-energy-for-the-future-1-detailed-eda @ 

  

from pandas.api.types import is_datetime64_any_dtype as is_datetime

from pandas.api.types import is_categorical_dtype





def reduce_mem_usage(df, use_float16=False):

    """

    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.        

    """

    

    start_mem = df.memory_usage().sum() / 1024**2

    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    

    for col in df.columns:

        if is_datetime(df[col]) or is_categorical_dtype(df[col]):

            continue

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == "int":

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype("category")

    

    end_mem = df.memory_usage().sum() / 1024**2

    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))

    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    

    return df
building_metadata = reduce_mem_usage(building_metadata,use_float16=True)

weather_train = reduce_mem_usage(weather_train,use_float16=True)

train = reduce_mem_usage(train,use_float16=True)



weather_test = reduce_mem_usage(weather_test,use_float16=True)

test = reduce_mem_usage(test,use_float16=True)
# Merge training dataset

train = train.merge(building_metadata, left_on='building_id',right_on='building_id',how='left')

train = train.merge(weather_train,how='left',left_on=['site_id','timestamp'],right_on=['site_id','timestamp'])

del weather_train



# Merge test dataset

test = test.merge(building_metadata, left_on='building_id',right_on='building_id',how='left')

test = test.merge(weather_test,how='left',left_on=['site_id','timestamp'],right_on=['site_id','timestamp'])

del weather_test
# Replace categorical values 



from sklearn.preprocessing import LabelEncoder



lb_make = LabelEncoder()



train["primary_use"] = lb_make.fit_transform(train["primary_use"])

test['primary_use']= lb_make.fit_transform(test["primary_use"])
# Drop columns

train = train.drop(['timestamp'], axis = 1)

test = test.drop(['timestamp'], axis = 1)
# Split train labels and training 



y_train = np.log1p(train["meter_reading"])

x_train = train.drop(['meter_reading'], axis = 1)

del train
# Replace NaN values

# from sklearn.impute import SimpleImputer



# imputer  = SimpleImputer(missing_values=np.nan, strategy='mean')



# x_train = imputer.fit_transform(x_train.values)

# test = imputer.fit_transform(test.values)
print(np.isnan(x_train).sum())

print(np.isnan(test).sum())
# Scale Data



# from sklearn.preprocessing import MinMaxScaler



# scaler = MinMaxScaler()

# scaler.fit(x_train.values)

# MinMaxScaler(copy=True, feature_range=(0, 1))
import tensorflow as tf

from tensorflow import keras



model = keras.Sequential([

    keras.layers.Dense(12, activation='relu'),

    keras.layers.Dense(1, kernel_initializer='normal')

])
model.compile(loss='mean_squared_error',optimizer=optimizer)

model.fit(x_train,y_train.values,epochs=10, batch_size = 100,validation_split = 1/5)
from sklearn.metrics import mean_squared_log_error



predictions = model.predict(x_train)

np.sqrt(mean_squared_log_error( y_train, predictions ))