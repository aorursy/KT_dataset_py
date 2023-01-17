import numpy as np

import pandas as pd

import os, gc, sys, warnings, random, math, psutil, pickle



from sklearn.preprocessing import LabelEncoder



warnings.filterwarnings('ignore')
print('Load Data')

train_df = pd.read_csv('../input/ashrae-energy-prediction/train.csv')

test_df = pd.read_csv('../input/ashrae-energy-prediction/test.csv')



building_df = pd.read_csv('../input/ashrae-energy-prediction/building_metadata.csv')



train_weather_df = pd.read_csv('../input/ashrae-energy-prediction/weather_train.csv')

test_weather_df = pd.read_csv('../input/ashrae-energy-prediction/weather_test.csv')
########################### Convert Timestamp to Date

#################################################################################

for df in [train_df, test_df, train_weather_df, test_weather_df]:

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    

for df in [train_df, test_df]:

    df['DT_Y'] = df['timestamp'].dt.year-2000

    df['DT_M'] = df['timestamp'].dt.month.astype(np.int8)

    df['DT_W'] = df['timestamp'].dt.weekofyear.astype(np.int8)

    df['DT_D'] = df['timestamp'].dt.dayofyear.astype(np.int16)

    df['DT_hour'] = df['timestamp'].dt.hour.astype(np.int8)

    df['DT_dayofweek'] = df['timestamp'].dt.dayofweek.astype(np.int8)

    df['DT_day_month'] = df['timestamp'].dt.day.astype(np.int8)

    df['DT_week_month'] = df['timestamp'].dt.day/7

    df['DT_week_month'] = df['DT_week_month'].apply(lambda x: math.ceil(x)).astype(np.int8)
########################### Strings to Category

#################################################################################

building_df['primary_use'] = building_df['primary_use'].astype('category')



########################### Building Transform

#################################################################################

building_df['floor_count'] = building_df['floor_count'].fillna(building_df['floor_count'].dropna().median()).astype(np.int8)

building_df['year_built'] = building_df['year_built'].fillna(building_df['year_built'].dropna().median()).astype(np.int16)



le = LabelEncoder()

building_df['primary_use'] = building_df['primary_use'].astype(str)

building_df['primary_use'] = le.fit_transform(building_df['primary_use']).astype(np.int8)
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
########################### Convert to Reduece Memory

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
########################### Building DF merge through concat 

#################################################################################

# Benefits of concat:

## Faster for huge datasets (columns number)

## No dtype change for dataset

## Consume less memmory 



temp_df = train_df[['building_id']]

temp_df = temp_df.merge(building_df, on=['building_id'], how='left')

del temp_df['building_id']

train_df = pd.concat([train_df, temp_df], axis=1)



temp_df = test_df[['building_id']]

temp_df = temp_df.merge(building_df, on=['building_id'], how='left')

del temp_df['building_id']

test_df = pd.concat([test_df, temp_df], axis=1)



del building_df, temp_df
########################### Weather DF merge over concat (to not lose type)

#################################################################################

# Benefits of concat:

## Faster for huge datasets (columns number)

## No dtype change for dataset

## Consume less memmory 



temp_df = train_df[['site_id','timestamp']]

temp_df = temp_df.merge(train_weather_df, on=['site_id','timestamp'], how='left')

del temp_df['site_id'], temp_df['timestamp']

train_df = pd.concat([train_df, temp_df], axis=1)



temp_df = test_df[['site_id','timestamp']]

temp_df = temp_df.merge(test_weather_df, on=['site_id','timestamp'], how='left')

del temp_df['site_id'], temp_df['timestamp']

test_df = pd.concat([test_df, temp_df], axis=1)



del train_weather_df, test_weather_df, temp_df

gc.collect()
def average_imputation(df, column_name):

    imputation = df.groupby(['timestamp'])[column_name].mean()

    

    df.loc[df[column_name].isnull(), column_name] = df[df[column_name].isnull()][[column_name]].apply(lambda x: imputation[df['timestamp'][x.index]].values)

    del imputation

    return df



beaufort = [(0, 0, 0.3), (1, 0.3, 1.6), (2, 1.6, 3.4), (3, 3.4, 5.5), (4, 5.5, 8), (5, 8, 10.8), (6, 10.8, 13.9), 

          (7, 13.9, 17.2), (8, 17.2, 20.8), (9, 20.8, 24.5), (10, 24.5, 28.5), (11, 28.5, 33), (12, 33, 200)]



class ASHRAE3Preprocessor(object):

    @classmethod

    def fit(cls, df):

        data_ratios = df.count()/len(df)

        cls.avgs = df.loc[:,data_ratios < 1.0].mean()



    @classmethod

    def transform(cls, df):

        #df = df.fillna(cls.avgs) # refill NAN with averages

        data_ratios = df.count()/len(df)

        columns_to_fill = data_ratios[data_ratios < 1.0].index.values.tolist()

        for col in columns_to_fill:

            df = average_imputation(df, col)

        

        for item in beaufort:

            df.loc[(df['wind_speed']>=item[1]) & (df['wind_speed']<item[2]), 'beaufort_scale'] = item[0]



        # parse and cast columns to a smaller type

        df.rename(columns={"square_feet": "log_square_feet"}, inplace=True)

        df['log_square_feet'] = np.float16(np.log(df['log_square_feet']))

        df['year_built'] = np.uint8(df['year_built']-1900)

        

        # remove redundant columns

        for col in df.columns:

            if col in ['timestamp', 'row_id', 'wind_speed']:

                del df[col]

    

        # extract target column

        if 'meter_reading' in df.columns:

            df['meter_reading'] = np.log1p(df['meter_reading']).astype(np.float32) # comp metric uses log errors

            # maybe remove some of the high outliers because of sensor error

            df["meter_reading"] = df["meter_reading"].clip(upper = df["meter_reading"].quantile(.999))



        return df

        

#ASHRAE3Preprocessor.fit(train_df)

train_df = ASHRAE3Preprocessor.transform(train_df)

test_df = ASHRAE3Preprocessor.transform(test_df)
########################### Trick to use kernel hdd to store results

#################################################################################

train_df.to_pickle('train_df.pkl')

test_df.to_pickle('test_df.pkl')

   

del train_df, test_df

gc.collect()