# General imports

import numpy as np

import pandas as pd

import os, warnings, math



from sklearn.preprocessing import LabelEncoder



warnings.filterwarnings('ignore')
########################### Helpers

#################################################################################

## -------------------

## Memory Reducer

# :df pandas dataframe to reduce size             # type: pd.DataFrame()

# :verbose                                        # type: bool

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df

## -------------------
########################### DATA LOAD

#################################################################################

print('Load Data')

train_df = pd.read_csv('../input/ashrae-energy-prediction/train.csv')

test_df = pd.read_csv('../input/ashrae-energy-prediction/test.csv')



building_df = pd.read_csv('../input/ashrae-energy-prediction/building_metadata.csv')



train_weather_df = pd.read_csv('../input/ashrae-energy-prediction/weather_train.csv')

test_weather_df = pd.read_csv('../input/ashrae-energy-prediction/weather_test.csv')
########################### Data Check

#################################################################################

print('Main data:', list(train_df), train_df.info())

print('#'*20)



print('Buildings data:',list(building_df), building_df.info())

print('#'*20)



print('Weather data:',list(train_weather_df), train_weather_df.info())

print('#'*20)
########################### Convert timestamp to date

#################################################################################

for df in [train_df, test_df, train_weather_df, test_weather_df]:

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    

for df in [train_df, test_df]:

    df['DT_M'] = df['timestamp'].dt.month.astype(np.int8)

    df['DT_W'] = df['timestamp'].dt.weekofyear.astype(np.int8)

    df['DT_D'] = df['timestamp'].dt.dayofyear.astype(np.int16)

    

    df['DT_hour'] = df['timestamp'].dt.hour.astype(np.int8)

    df['DT_day_week'] = df['timestamp'].dt.dayofweek.astype(np.int8)

    df['DT_day_month'] = df['timestamp'].dt.day.astype(np.int8)

    df['DT_week_month'] = df['timestamp'].dt.day/7

    df['DT_week_month'] = df['DT_week_month'].apply(lambda x: math.ceil(x)).astype(np.int8)
########################### Strings to category

#################################################################################

building_df['primary_use'] = building_df['primary_use'].astype('category')
########################### Building Transform

#################################################################################

building_df['floor_count'] = building_df['floor_count'].fillna(0).astype(np.int8)

building_df['year_built'] = building_df['year_built'].fillna(-999).astype(np.int16)



le = LabelEncoder()

building_df['primary_use'] = building_df['primary_use'].astype(str)

building_df['primary_use'] = le.fit_transform(building_df['primary_use']).astype(np.int8)
########################### Base check

#################################################################################

do_not_convert = ['category','datetime64[ns]','object']

for df in [train_df, test_df, building_df, train_weather_df, test_weather_df]:

    original = df.copy()

    df = reduce_mem_usage(df)



    for col in list(df):

        if df[col].dtype.name not in do_not_convert:

            if (df[col]-original[col]).sum()!=0:

                df[col] = original[col]

                print('Bad transformation', col)
########################### Data Check

#################################################################################

print('Main data:', list(train_df), train_df.info())

print('#'*20)



print('Buildings data:',list(building_df), building_df.info())

print('#'*20)



print('Weather data:',list(train_weather_df), train_weather_df.info())

print('#'*20)
########################### Export (using same names as in competition dataset)

#################################################################################

train_df.to_pickle('train.pkl')

test_df.to_pickle('test.pkl')



building_df.to_pickle('building_metadata.pkl')



train_weather_df.to_pickle('weather_train.pkl')

test_weather_df.to_pickle('weather_test.pkl')