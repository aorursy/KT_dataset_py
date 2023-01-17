import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc



# matplotlib and seaborn for plotting

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

import matplotlib.patches as patches



from plotly import tools, subplots

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px

pd.set_option('max_columns', 150)



py.init_notebook_mode(connected=True)

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



import os,random, math, psutil, pickle    
%%time

root = 'ashrae-energy-prediction/'

train_df = pd.read_csv(root + 'train.csv')

train_df["timestamp"] = pd.to_datetime(train_df["timestamp"], format='%Y-%m-%d %H:%M:%S')



weather_train_df = pd.read_csv(root + 'weather_train.csv')

test_df = pd.read_csv(root + 'test.csv')

weather_test_df = pd.read_csv(root + 'weather_test.csv')

building_meta_df = pd.read_csv(root + 'building_metadata.csv')

sample_submission = pd.read_csv(root + 'sample_submission.csv')
print('Size of train_df data', train_df.shape)

print('Size of weather_train_df data', weather_train_df.shape)

print('Size of weather_test_df data', weather_test_df.shape)

print('Size of building_meta_df data', building_meta_df.shape)
## Function to reduce the DF size

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
train_df = reduce_mem_usage(train_df)

test_df = reduce_mem_usage(test_df)



weather_train_df = reduce_mem_usage(weather_train_df)

weather_test_df = reduce_mem_usage(weather_test_df)

building_meta_df = reduce_mem_usage(building_meta_df)
train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])

test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])

weather_train_df['timestamp'] = pd.to_datetime(weather_train_df['timestamp'])

weather_test_df['timestamp'] = pd.to_datetime(weather_test_df['timestamp'])

    

building_meta_df['primary_use'] = building_meta_df['primary_use'].astype('category')
   

temp_df = train_df[['building_id']]

temp_df = temp_df.merge(building_meta_df, on=['building_id'], how='left')

del temp_df['building_id']

train_df = pd.concat([train_df, temp_df], axis=1)



temp_df = test_df[['building_id']]

temp_df = temp_df.merge(building_meta_df, on=['building_id'], how='left')



del temp_df['building_id']

test_df = pd.concat([test_df, temp_df], axis=1)

del temp_df, building_meta_df

temp_df = train_df[['site_id','timestamp']]

temp_df = temp_df.merge(weather_train_df, on=['site_id','timestamp'], how='left')



del temp_df['site_id'], temp_df['timestamp']

train_df = pd.concat([train_df, temp_df], axis=1)



temp_df = test_df[['site_id','timestamp']]

temp_df = temp_df.merge(weather_test_df, on=['site_id','timestamp'], how='left')



del temp_df['site_id'], temp_df['timestamp']

test_df = pd.concat([test_df, temp_df], axis=1)



del temp_df, weather_train_df, weather_test_df
train_df.to_pickle('train_df.pkl')

test_df.to_pickle('test_df.pkl')

   

del train_df, test_df

gc.collect()
train_df = pd.read_pickle('train_df.pkl')

test_df = pd.read_pickle('test_df.pkl')
train_df['age'] = train_df['year_built'].max() - train_df['year_built'] + 1

test_df['age'] = test_df['year_built'].max() - test_df['year_built'] + 1


train_df['floor_count'] = train_df['floor_count'].fillna(-999).astype(np.int16)

test_df['floor_count'] = test_df['floor_count'].fillna(-999).astype(np.int16)



train_df['year_built'] = train_df['year_built'].fillna(-999).astype(np.int16)

test_df['year_built'] = test_df['year_built'].fillna(-999).astype(np.int16)



train_df['age'] = train_df['age'].fillna(-999).astype(np.int16)

test_df['age'] = test_df['age'].fillna(-999).astype(np.int16)



train_df['cloud_coverage'] = train_df['cloud_coverage'].fillna(-999).astype(np.int16)

test_df['cloud_coverage'] = test_df['cloud_coverage'].fillna(-999).astype(np.int16) 

train_df['month_datetime'] = train_df['timestamp'].dt.month.astype(np.int8)

train_df['weekofyear_datetime'] = train_df['timestamp'].dt.weekofyear.astype(np.int8)

train_df['dayofyear_datetime'] = train_df['timestamp'].dt.dayofyear.astype(np.int16)

    

train_df['hour_datetime'] = train_df['timestamp'].dt.hour.astype(np.int8)  

train_df['day_week'] = train_df['timestamp'].dt.dayofweek.astype(np.int8)

train_df['day_month_datetime'] = train_df['timestamp'].dt.day.astype(np.int8)

train_df['week_month_datetime'] = train_df['timestamp'].dt.day/7

train_df['week_month_datetime'] = train_df['week_month_datetime'].apply(lambda x: math.ceil(x)).astype(np.int8)

    

train_df['year_built'] = train_df['year_built']-1900

train_df['square_feet'] = np.log(train_df['square_feet'])

    

test_df['month_datetime'] = test_df['timestamp'].dt.month.astype(np.int8)

test_df['weekofyear_datetime'] = test_df['timestamp'].dt.weekofyear.astype(np.int8)

test_df['dayofyear_datetime'] = test_df['timestamp'].dt.dayofyear.astype(np.int16)

    

test_df['hour_datetime'] = test_df['timestamp'].dt.hour.astype(np.int8)

test_df['day_week'] = test_df['timestamp'].dt.dayofweek.astype(np.int8)

test_df['day_month_datetime'] = test_df['timestamp'].dt.day.astype(np.int8)

test_df['week_month_datetime'] = test_df['timestamp'].dt.day/7

test_df['week_month_datetime'] = test_df['week_month_datetime'].apply(lambda x: math.ceil(x)).astype(np.int8)

    

test_df['year_built'] = test_df['year_built']-1900

test_df['square_feet'] = np.log(test_df['square_feet'])
# The IDs of the notorious 22 buildings, taken from the graphs in  CaesarLupum's notebook 

ids = [803,801,799,1088,993,794,1284,76,1258,1289,778,29,1156,29,60,50,1099,1197,1168,1148,1021,1221]
# copy of the train and test dataframes 

train_copy = train_df.copy(deep=True)

test_copy = test_df.copy(deep=True)
to_drop = ['precip_depth_1_hr',

 'cloud_coverage',

 'wind_direction',

'primary_use' ,                            

'year_built',       

'weekofyear_datetime' ,

'month_datetime' ,   

'floor_count',          

'sea_level_pressure',   

'dew_temperature',       

'cloud_coverage',        

'day_week',              

'wind_direction',        

'day_month_datetime',    

'week_month_datetime',  

 ]
# We drop the previous columns and remove the 22 buildings

train_copy = train_copy.drop(columns=to_drop,axis=1)

train_copy = train_copy[~train_copy1.building_id.isin(ids)]

# It was a headache to feed this value to the model so we dropped it and prayed that the day and the hour would be enough

train_copy = train_copy.drop(['timestamp'],axis=1)

test_copy = test_copy.drop(['timestamp'],axis=1)
train_copy['square_feet']=(train_copy['square_feet']-train_copy['square_feet'].min())/(train_copy['square_feet'].max()-train_copy['square_feet'].min())

train_copy['air_temperature']=(train_copy['air_temperature']-train_copy['air_temperature'].min())/(train_copy['air_temperature'].max()-train_copy['air_temperature'].min())

train_copy['age']=(train_copy['age']-train_copy['age'].min())/(train_copy['age'].max()-train_copy['age'].min())

train_copy['wind_speed']=(train_copy['wind_speed']-train_copy['wind_speed'].min())/(train_copy['wind_speed'].max()-train_copy['wind_speed'].min())
shuffled_train = train_copy.sample(frac=1)
y_train=shuffled_train['meter_reading']

shuffled_train= shuffled_train.drop(['meter_reading'],axis=1)
test_copy = test_copy.drop(columns=to_drop,axis=1)
# save the row id separately

row_id = test_copy['row_id']

test_copy = test_copy.drop(['row_id'],axis=1)
test_copy['square_feet']=(test_copy['square_feet']-test_copy['square_feet'].min())/(test_copy['square_feet'].max()-test_copy['square_feet'].min())

test_copy['air_temperature']=(test_copy['air_temperature']-test_copy['air_temperature'].min())/(test_copy['air_temperature'].max()-test_copy['air_temperature'].min())

test_copy['age']=(test_copy['age']-test_copy['age'].min())/(test_copy['age'].max()-test_copy['age'].min())

test_copy['wind_speed']=(test_copy['wind_speed']-test_copy['wind_speed'].min())/(test_copy['wind_speed'].max()-test_copy['wind_speed'].min())

test_copy['dayofyear_datetime']=(test_copy['dayofyear_datetime']-test_copy['dayofyear_datetime'].min())/(test_copy['dayofyear_datetime'].max()-test_copy['dayofyear_datetime'].min())

test_copy['hour_datetime']=(test_copy['hour_datetime']-test_copy['hour_datetime'].min())/(test_copy['hour_datetime'].max()-test_copy['hour_datetime'].min())
import xgboost as xgb

from sklearn.metrics import mean_squared_error
xgboost_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,

                max_depth = 5, alpha = 10, n_estimators = 10)
# Fitting

xgboost_reg.fit(shuffled_train,y_train)
# Predicting

preds = xgboost_reg.predict(test_copy)
# Writing the results into a file 

dt = pd.DataFrame(data=preds)

dt.to_csv('test_try_xgb.csv',  index=True)