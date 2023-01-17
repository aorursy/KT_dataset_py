# General imports

import numpy as np

import pandas as pd

import os, warnings, gc, math



pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)



warnings.filterwarnings('ignore')

NROWS = None
# Memory reducer function

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
%%time

# Data load

train_df = pd.read_csv('../input/ashrae-energy-prediction/train.csv', nrows = NROWS)

test_df = pd.read_csv('../input/ashrae-energy-prediction/test.csv', nrows = NROWS)



building_df = pd.read_csv('../input/ashrae-energy-prediction/building_metadata.csv', nrows = NROWS)



train_weather_df = pd.read_csv('../input/ashrae-energy-prediction/weather_train.csv', nrows = NROWS)

test_weather_df = pd.read_csv('../input/ashrae-energy-prediction/weather_test.csv', nrows = NROWS)
%%time

# Date convertions

for df in [train_df, test_df, train_weather_df, test_weather_df]:

    

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    

# for df in [train_df, test_df, train_weather_df, test_weather_df]:

for df in [train_df, test_df]:

    df['hour'] = np.uint8(df['timestamp'].dt.hour)

    df['day'] = np.uint8(df['timestamp'].dt.day)

    df['weekday'] = np.uint8(df['timestamp'].dt.weekday)

    df['month'] = np.uint8(df['timestamp'].dt.month)

    df['year'] = np.uint8(df['timestamp'].dt.year-2000)

    

# Categorical convertions

%%time

# Fill NA

building_df.rename(columns={"square_feet": "log_square_feet"}, inplace=True)

building_df['log_square_feet'] = np.float16(np.log(building_df['log_square_feet']))

building_df['year_built'] = np.uint8(building_df['year_built']-1900)

building_df['floor_count'] = np.uint8(building_df['floor_count'])



# Enconding

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

building_df['primary_use'] = building_df['primary_use'].astype(str)

building_df['primary_use'] = le.fit_transform(building_df['primary_use']).astype(np.int8)
%%time

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer



for df in [train_weather_df, test_weather_df]:

    cols = list(df.columns)

    cols.remove('timestamp')

    imp = IterativeImputer(random_state=42)

    temp = imp.fit_transform(df[cols])

    df[cols] = pd.DataFrame(temp, columns = cols)
%%time

for df in [train_df, test_df, building_df, train_weather_df, test_weather_df]:

    original = df.copy()

    df = reduce_mem_usage(df)
%%time

temp_df = train_df[['building_id']]

temp_df = temp_df.merge(building_df, on=['building_id'], how='left')

del temp_df['building_id']

train_df = pd.concat([train_df, temp_df], axis=1)

del temp_df
%%time

temp_df = test_df[['building_id']]

temp_df = temp_df.merge(building_df, on=['building_id'], how='left')

del temp_df['building_id']

test_df = pd.concat([test_df, temp_df], axis=1)

del temp_df
%%time

temp_df = train_df[['site_id','timestamp']]

temp_df = temp_df.merge(train_weather_df, on=['site_id','timestamp'], how='left')

del temp_df['site_id'], temp_df['timestamp']

train_df = pd.concat([train_df, temp_df], axis=1)

del temp_df
%%time

temp_df = test_df[['site_id','timestamp']]

temp_df = temp_df.merge(test_weather_df, on=['site_id','timestamp'], how='left')

del temp_df['site_id'], temp_df['timestamp']

test_df = pd.concat([test_df, temp_df], axis=1)



%%time

del train_weather_df, test_weather_df, temp_df

gc.collect()
%%time

for m in train_df.meter.unique():

#     train_df[train_df.meter == m].to_parquet('train'+ str(m) + '.parquet')

#     test_df[test_df.meter == m].to_parquet('test'+ str(m) + '.parquet')

    train_df[train_df.meter == m].to_pickle('train'+ str(m) + '.pkl')

    test_df[test_df.meter == m].to_pickle('test'+ str(m) + '.pkl')
