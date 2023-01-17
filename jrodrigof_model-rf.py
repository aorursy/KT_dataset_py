import numpy as np

import pandas as pd 

import gc



import matplotlib.pyplot as plt

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



import os

import random

import math

import psutil

import pickle

import timeit



from sklearn.model_selection import train_test_split,KFold

from sklearn.preprocessing import LabelEncoder



from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"



from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer

import pandas_profiling



%%time



metadata_dtype = {'site_id':"uint8",'building_id':'uint16','square_feet':'float32','year_built':'float32','floor_count':"float16"}

weather_dtype = {"site_id":"uint8",'air_temperature':"float16",'cloud_coverage':"float16",'dew_temperature':"float16",'precip_depth_1_hr':"float16",

                 'sea_level_pressure':"float32",'wind_direction':"float16",'wind_speed':"float16"}

train_dtype = {'meter':"uint8",'building_id':'uint16','meter_reading':"float32"}



start_time = timeit.default_timer()



weather_train = pd.read_csv("../input/ashrae-energy-prediction/weather_train.csv", parse_dates=['timestamp'], dtype=weather_dtype)

# weather_test = pd.read_csv("../input/ashrae-energy-prediction/weather_test.csv", parse_dates=['timestamp'], dtype=weather_dtype)



metadata = pd.read_csv("../input/ashrae-energy-prediction/building_metadata.csv", dtype=metadata_dtype)



train = pd.read_csv("../input/ashrae-energy-prediction/train.csv", parse_dates=['timestamp'], dtype=train_dtype)

# test = pd.read_csv("../input/ashrae-energy-prediction/test.csv", parse_dates=['timestamp'], usecols=['building_id','meter','timestamp'], dtype=train_dtype)



print('Size of train_df data', train.shape)

print('Size of weather_train_df data', weather_train.shape)

# print('Size of weather_test_df data', weather_test.shape)

print('Size of building_meta_df data', metadata.shape)



elapsed = timeit.default_timer() - start_time

print(elapsed)

weather_train.head()

metadata.head()

train.head()
weather_test = pd.read_csv("weather_test.csv", parse_dates=['timestamp'], dtype=weather_dtype)

print('Size of weather_test_df data', weather_test.shape)



cols = list(weather_train.columns[2:])

cols_imputed = weather_train[cols].isnull().astype('bool_').add_suffix('_imputed')



imp = IterativeImputer(max_iter=10, verbose=0)

imp.fit(weather_train.iloc[:,2:])

weather_train_imputed = imp.transform(weather_train.iloc[:,2:])

weather_train_imputed = pd.concat([weather_train.iloc[:,0:2],pd.DataFrame(weather_train_imputed, columns=weather_train.columns[2:]), cols_imputed], axis=1)

# weather_train_imputed = pd.concat([weather_train.iloc[:,0:2],pd.DataFrame(weather_train_imputed, columns=weather_train.columns[2:])], axis=1)

pd.DataFrame(weather_train_imputed.isna().sum()/len(weather_train_imputed),columns=["Weather_Train_Missing_Imputed"])



# cols = list(weather_test.columns[2:])

# cols_imputed = weather_test[cols].isnull().astype('bool_').add_suffix('_imputed')



# imp = IterativeImputer(max_iter=10, verbose=0)

# imp.fit(weather_test.iloc[:,2:])

# weather_test_imputed = imp.transform(weather_test.iloc[:,2:])

# weather_test_imputed = pd.concat([weather_test.iloc[:,0:2],pd.DataFrame(weather_test_imputed, columns=weather_test.columns[2:]), cols_imputed], axis=1)

# weather_test_imputed = pd.concat([weather_test.iloc[:,0:2],pd.DataFrame(weather_test_imputed, columns=weather_test.columns[2:])], axis=1)

# pd.DataFrame(weather_test_imputed.isna().sum()/len(weather_test_imputed),columns=["Weather_Train_Missing_Imputed"])



# imputation floor_count & year built



cols = list(metadata.columns[4:])

cols_imputed = metadata[cols].isnull().astype('uint8').add_suffix('_imputed')



imp = IterativeImputer(max_iter=10, verbose=0)

imp.fit(metadata.iloc[:,3:])

metadata_imputed = imp.transform(metadata.iloc[:,3:])

metadata_imputed = pd.concat([metadata.iloc[:,0:3],pd.DataFrame(metadata_imputed, columns=metadata.columns[3:]), cols_imputed], axis=1)

#metadata_imputed = pd.concat([metadata.iloc[:,0:3],pd.DataFrame(metadata_imputed, columns=metadata.columns[3:])], axis=1)

pd.DataFrame(metadata_imputed.isna().sum()/len(metadata_imputed),columns=["Metadata_Missing_Imputed"])



metadata_imputed.year_built = metadata_imputed.year_built.round()

metadata_imputed.floor_count = metadata_imputed.floor_count.round()



del weather_train

# del weather_test

del metadata

del cols

del cols_imputed



gc.collect()

# for df in [train, test]:

for df in [train]:

    df['Month'] = df['timestamp'].dt.month.astype("uint8")

    df['DayOfMonth'] = df['timestamp'].dt.day.astype("uint8")

    df['DayOfWeek'] = df['timestamp'].dt.dayofweek.astype("uint8")

    df['Hour'] = df['timestamp'].dt.hour.astype("uint8")

    df['timestamp_2'] = (df.timestamp - pd.to_datetime("2016-01-01")).dt.total_seconds() // 3600

    df['timestamp_2'] = df.timestamp_2.astype('uint16')

    

# Code to read and combine the standard input files, converting timestamps to number of hours since the beginning of 2016.



weather_train_imputed['timestamp_2'] = (weather_train_imputed.timestamp - pd.to_datetime("2016-01-01")).dt.total_seconds() // 3600

weather_train_imputed['timestamp_2'] = weather_train_imputed.timestamp_2.astype('int16')



# weather_test_imputed['timestamp_2'] = (weather_test_imputed.timestamp - pd.to_datetime("2016-01-01")).dt.total_seconds() // 3600

# weather_test_imputed['timestamp_2'] = weather_test_imputed.timestamp_2.astype('int16')



#fix timestamps



site_GMT_offsets = [-5, 0, -7, -5, -8, 0, -5, -5, -5, -6, -7, -5, 0, -6, -5, -5]

GMT_offset_map = {site: offset for site, offset in enumerate(site_GMT_offsets)}

weather_train_imputed.timestamp_2 = weather_train_imputed.timestamp_2 + weather_train_imputed.site_id.map(GMT_offset_map)

# weather_test_imputed.timestamp_2 = weather_test_imputed.timestamp_2 + weather_test_imputed.site_id.map(GMT_offset_map)



weather_train_imputed.drop('timestamp',axis=1,inplace=True)

# weather_test_imputed.drop('timestamp',axis=1,inplace=True)

gc.collect()



# Dropping floor_count variable as it has 75% missing values

# metadata_imputed.drop('floor_count',axis=1,inplace=True)



#train['meter'].replace({0:"Electricity",1:"ChilledWater",2:"Steam",3:"HotWater"},inplace=True)

#test['meter'].replace({0:"Electricity",1:"ChilledWater",2:"Steam",3:"HotWater"},inplace=True)



train.rename(columns={'timestamp':'timestamp_train'}, inplace=True)

# test.rename(columns={'timestamp':'timestamp_test'}, inplace=True)



train['meter_reading'] = np.log1p(train['meter_reading'])



metadata_imputed['primary_use'].replace({"Healthcare":"Other","Parking":"Other","Warehouse/storage":"Other","Manufacturing/industrial":"Other",

                                "Retail":"Other","Services":"Other","Technology/science":"Other","Food sales and service":"Other",

                                "Utility":"Other","Religious worship":"Other"},inplace=True)

metadata_imputed['square_feet'] = np.log1p(metadata_imputed['square_feet'])

# metadata_imputed['year_built'].fillna(-999, inplace=True)

metadata_imputed['square_feet'] = metadata_imputed['square_feet'].astype('float16')

metadata_imputed['year_built'] = metadata_imputed['year_built'].astype('uint16')

metadata_imputed['floor_count'] = metadata_imputed['floor_count'].astype('uint8')



gc.collect()

%%time

train = pd.merge(train,metadata_imputed,on='building_id',how='left')

# test  = pd.merge(test,metadata_imputed,on='building_id',how='left')

print ("Training Data+Metadata Shape {}".format(train.shape))

# print ("Testing Data+Metadata Shape {}".format(test.shape))

del metadata_imputed

gc.collect()



train = pd.merge(train,weather_train_imputed,on=['site_id','timestamp_2'],how='left')

del weather_train_imputed

gc.collect()



# test  = pd.merge(test,weather_test_imputed,on=['site_id','timestamp_2'],how='left')

print ("Training Data+Metadata+Weather Shape {}".format(train.shape))

# print ("Testing Data+Metadata+Weather Shape {}".format(test.shape))



# del weather_test_imputed

# gc.collect()



#missing_train = pd.DataFrame(train.isna().sum()/len(train),columns=["Train_Missing"])

#missing_train



#missing_test = pd.DataFrame(test.isna().sum()/len(test),columns=["Train_Missing"])

#missing_test



# Save space

# commented since already done above

#for df in [train,test]:

#    df['square_feet'] = df['square_feet'].astype('float16')

    

# Fill NA



#cols = ['air_temperature','cloud_coverage','dew_temperature','precip_depth_1_hr','sea_level_pressure','wind_direction','wind_speed']

#for col in cols:

#    train[col].fillna(train[col].mean(),inplace=True)

#    test[col].fillna(test[col].mean(),inplace=True)

    

# Drop nonsense entries

# As per the discussion in the following thread, https://www.kaggle.com/c/ashrae-energy-prediction/discussion/117083, there is some discrepancy in the meter_readings for different ste_id's and buildings. It makes sense to delete them

idx_to_drop = list((train[(train['site_id'] == 0) & (train['timestamp_train'] < "2016-05-21 00:00:00")]).index)

print (len(idx_to_drop))

train.drop(idx_to_drop,axis='rows',inplace=True)



# dropping all the electricity meter readings that are 0, after considering them as anomalies.

# idx_to_drop = list(train[(train['meter'] == "Electricity") & (train['meter_reading'] == 0)].index)

idx_to_drop = list(train[(train['meter'] == 0) & (train['meter_reading'] == 0)].index)

print(len(idx_to_drop))

train.drop(idx_to_drop,axis='rows',inplace=True)



##train.drop('timestamp',axis=1,inplace=True)

##test.drop('timestamp',axis=1,inplace=True)



train.drop('timestamp_train',axis=1,inplace=True)

# test.drop('timestamp_test',axis=1,inplace=True)



train.drop('timestamp_2',axis=1,inplace=True)

# test.drop('timestamp_2',axis=1,inplace=True)



del idx_to_drop

gc.collect()

%%time

number_unique_meter_per_building = train.groupby('building_id')['meter'].nunique()

train['number_unique_meter_per_building'] = train['building_id'].map(number_unique_meter_per_building)



mean_meter_reading_per_building = train.groupby('building_id')['meter_reading'].mean()

train['mean_meter_reading_per_building'] = train['building_id'].map(mean_meter_reading_per_building)

median_meter_reading_per_building = train.groupby('building_id')['meter_reading'].median()

train['median_meter_reading_per_building'] = train['building_id'].map(median_meter_reading_per_building)

std_meter_reading_per_building = train.groupby('building_id')['meter_reading'].std()

train['std_meter_reading_per_building'] = train['building_id'].map(std_meter_reading_per_building)





mean_meter_reading_on_year_built = train.groupby('year_built')['meter_reading'].mean()

train['mean_meter_reading_on_year_built'] = train['year_built'].map(mean_meter_reading_on_year_built)

median_meter_reading_on_year_built = train.groupby('year_built')['meter_reading'].median()

train['median_meter_reading_on_year_built'] = train['year_built'].map(median_meter_reading_on_year_built)

std_meter_reading_on_year_built = train.groupby('year_built')['meter_reading'].std()

train['std_meter_reading_on_year_built'] = train['year_built'].map(std_meter_reading_on_year_built)





mean_meter_reading_per_meter = train.groupby('meter')['meter_reading'].mean()

train['mean_meter_reading_per_meter'] = train['meter'].map(mean_meter_reading_per_meter)

median_meter_reading_per_meter = train.groupby('meter')['meter_reading'].median()

train['median_meter_reading_per_meter'] = train['meter'].map(median_meter_reading_per_meter)

std_meter_reading_per_meter = train.groupby('meter')['meter_reading'].std()

train['std_meter_reading_per_meter'] = train['meter'].map(std_meter_reading_per_meter)





mean_meter_reading_per_primary_usage = train.groupby('primary_use')['meter_reading'].mean()

train['mean_meter_reading_per_primary_usage'] = train['primary_use'].map(mean_meter_reading_per_primary_usage)

median_meter_reading_per_primary_usage = train.groupby('primary_use')['meter_reading'].median()

train['median_meter_reading_per_primary_usage'] = train['primary_use'].map(median_meter_reading_per_primary_usage)

std_meter_reading_per_primary_usage = train.groupby('primary_use')['meter_reading'].std()

train['std_meter_reading_per_primary_usage'] = train['primary_use'].map(std_meter_reading_per_primary_usage)





mean_meter_reading_per_site_id = train.groupby('site_id')['meter_reading'].mean()

train['mean_meter_reading_per_site_id'] = train['site_id'].map(mean_meter_reading_per_site_id)

median_meter_reading_per_site_id = train.groupby('site_id')['meter_reading'].median()

train['median_meter_reading_per_site_id'] = train['site_id'].map(median_meter_reading_per_site_id)

std_meter_reading_per_site_id = train.groupby('site_id')['meter_reading'].std()

train['std_meter_reading_per_site_id'] = train['site_id'].map(std_meter_reading_per_site_id)





# test['number_unique_meter_per_building'] = test['building_id'].map(number_unique_meter_per_building)



# test['mean_meter_reading_per_building'] = test['building_id'].map(mean_meter_reading_per_building)

# test['median_meter_reading_per_building'] = test['building_id'].map(median_meter_reading_per_building)

# test['std_meter_reading_per_building'] = test['building_id'].map(std_meter_reading_per_building)



# test['mean_meter_reading_on_year_built'] = test['year_built'].map(mean_meter_reading_on_year_built)

# test['median_meter_reading_on_year_built'] = test['year_built'].map(median_meter_reading_on_year_built)

# test['std_meter_reading_on_year_built'] = test['year_built'].map(std_meter_reading_on_year_built)



# test['mean_meter_reading_per_meter'] = test['meter'].map(mean_meter_reading_per_meter)

# test['median_meter_reading_per_meter'] = test['meter'].map(median_meter_reading_per_meter)

# test['std_meter_reading_per_meter'] = test['meter'].map(std_meter_reading_per_meter)



# test['mean_meter_reading_per_primary_usage'] = test['primary_use'].map(mean_meter_reading_per_primary_usage)

# test['median_meter_reading_per_primary_usage'] = test['primary_use'].map(median_meter_reading_per_primary_usage)

# test['std_meter_reading_per_primary_usage'] = test['primary_use'].map(std_meter_reading_per_primary_usage)



# test['mean_meter_reading_per_site_id'] = test['site_id'].map(mean_meter_reading_per_site_id)

# test['median_meter_reading_per_site_id'] = test['site_id'].map(median_meter_reading_per_site_id)

# test['std_meter_reading_per_site_id'] = test['site_id'].map(std_meter_reading_per_site_id)
%%time

# for df in [train, test]:

for df in [train]:

    df['mean_meter_reading_per_building'] = df['mean_meter_reading_per_building'].astype("float16")

    df['median_meter_reading_per_building'] = df['mean_meter_reading_per_building'].astype("float16")

    df['std_meter_reading_per_building'] = df['std_meter_reading_per_building'].astype("float16")

    

    df['mean_meter_reading_on_year_built'] = df['mean_meter_reading_on_year_built'].astype("float16")

    df['median_meter_reading_on_year_built'] = df['median_meter_reading_on_year_built'].astype("float16")

    df['std_meter_reading_on_year_built'] = df['std_meter_reading_on_year_built'].astype("float16")

    

    df['mean_meter_reading_per_meter'] = df['mean_meter_reading_per_meter'].astype("float16")

    df['median_meter_reading_per_meter'] = df['median_meter_reading_per_meter'].astype("float16")

    df['std_meter_reading_per_meter'] = df['std_meter_reading_per_meter'].astype("float16")

    

    df['mean_meter_reading_per_primary_usage'] = df['mean_meter_reading_per_primary_usage'].astype("float16")

    df['median_meter_reading_per_primary_usage'] = df['median_meter_reading_per_primary_usage'].astype("float16")

    df['std_meter_reading_per_primary_usage'] = df['std_meter_reading_per_primary_usage'].astype("float16")

    

    df['mean_meter_reading_per_site_id'] = df['mean_meter_reading_per_site_id'].astype("float16")

    df['median_meter_reading_per_site_id'] = df['median_meter_reading_per_site_id'].astype("float16")

    df['std_meter_reading_per_site_id'] = df['std_meter_reading_per_site_id'].astype("float16")

    

    df['number_unique_meter_per_building'] = df['number_unique_meter_per_building'].astype('uint8')

gc.collect()
le = LabelEncoder()



train['meter']= le.fit_transform(train['meter']).astype("uint8")

# test['meter']= le.fit_transform(test['meter']).astype("uint8")

train['primary_use']= le.fit_transform(train['primary_use']).astype("uint8")

# test['primary_use']= le.fit_transform(test['primary_use']).astype("uint8")



# print (train.shape, test.shape)

print (train.shape)
%%time

# Let's check the correlation between the variables and eliminate the one's that have high correlation

# Threshold for removing correlated variables

threshold = 0.9



# Absolute value correlation matrix

corr_matrix = train.corr().abs()

# Upper triangle of correlations

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

del corr_matrix

gc.collect()



# Select columns with correlations above threshold

to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

del upper 

gc.collect()



print('There are %d columns to remove.' % (len(to_drop)))

print ("Following columns can be dropped {}".format(to_drop))



train.drop(to_drop,axis=1,inplace=True)

# test.drop(to_drop,axis=1,inplace=True)



# del to_drop

gc.collect()
%%time

y = train['meter_reading']

train.drop('meter_reading',axis=1,inplace=True)

categorical_cols = ['building_id','Month','meter','Hour','primary_use','DayOfWeek','DayOfMonth']
meter_cut, bins = pd.cut(y, bins=50, retbins=True)

meter_cut.value_counts()
x_train,x_test,y_train,y_test = train_test_split(train,y,test_size=0.2,random_state=42, stratify=meter_cut)

print (x_train.shape)

print (y_train.shape)

print (x_test.shape)

print (y_test.shape)



train_columns = train.columns

del train

del meter_cut

del bins

gc.collect()
from sklearn.ensemble import RandomForestRegressor as RF

import lightgbm as lgb
lgb_train = lgb.Dataset(x_train, y_train,categorical_feature=categorical_cols)

del x_train, y_train

gc.collect() 



lgb_test = lgb.Dataset(x_test, y_test,categorical_feature=categorical_cols)

del x_test, y_test

gc.collect() 



params = {'feature_fraction': 0.75,

          'bagging_fraction': 0.75,

          'objective': 'regression',

          'max_depth': -1,

          'learning_rate': 0.15,

          "boosting_type": "gbdt",

          "bagging_seed": 11,

          "metric": 'rmse',

          "verbosity": -1,

          'reg_alpha': 0.5,

          'reg_lambda': 0.5,

          'random_state': 47

         }



reg = lgb.train(params, lgb_train, num_boost_round=3000, valid_sets=[lgb_train, lgb_test], early_stopping_rounds=100, verbose_eval = 100)



del lgb_train, lgb_test

gc.collect() 

# ser = pd.DataFrame(reg.feature_importance(),train.columns,columns=['Importance']).sort_values(by='Importance')

ser = pd.DataFrame(reg.feature_importance(),train_columns,columns=['Importance']).sort_values(by='Importance')

ser['Importance'].plot(kind='bar',figsize=(10,6))



#del train

del ser

del train_columns

gc.collect() 

# loading and processing of test objects



weather_test = pd.read_csv("weather_test.csv", parse_dates=['timestamp'], dtype=weather_dtype)

print('Size of weather_test_df data', weather_test.shape)



cols = list(weather_test.columns[2:])

cols_imputed = weather_test[cols].isnull().astype('bool_').add_suffix('_imputed')



imp = IterativeImputer(max_iter=10, verbose=0)

imp.fit(weather_test.iloc[:,2:])

weather_test_imputed = imp.transform(weather_test.iloc[:,2:])

weather_test_imputed = pd.concat([weather_test.iloc[:,0:2],pd.DataFrame(weather_test_imputed, columns=weather_test.columns[2:]), cols_imputed], axis=1)

weather_test_imputed = pd.concat([weather_test.iloc[:,0:2],pd.DataFrame(weather_test_imputed, columns=weather_test.columns[2:])], axis=1)

pd.DataFrame(weather_test_imputed.isna().sum()/len(weather_test_imputed),columns=["Weather_Train_Missing_Imputed"])



del cols

del cols_imputed

del weather_test

gc.collect()



test = pd.read_csv("test.csv", parse_dates=['timestamp'], usecols=['building_id','meter','timestamp'], dtype=train_dtype)



#

for df in [test]:

    df['Month'] = df['timestamp'].dt.month.astype("uint8")

    df['DayOfMonth'] = df['timestamp'].dt.day.astype("uint8")

    df['DayOfWeek'] = df['timestamp'].dt.dayofweek.astype("uint8")

    df['Hour'] = df['timestamp'].dt.hour.astype("uint8")

    df['timestamp_2'] = (df.timestamp - pd.to_datetime("2016-01-01")).dt.total_seconds() // 3600

    df['timestamp_2'] = df.timestamp_2.astype('uint16')



site_GMT_offsets = [-5, 0, -7, -5, -8, 0, -5, -5, -5, -6, -7, -5, 0, -6, -5, -5]

GMT_offset_map = {site: offset for site, offset in enumerate(site_GMT_offsets)}



weather_test_imputed.timestamp_2 = weather_test_imputed.timestamp_2 + weather_test_imputed.site_id.map(GMT_offset_map)

weather_test_imputed.drop('timestamp',axis=1,inplace=True)



#

test.rename(columns={'timestamp':'timestamp_test'}, inplace=True)



#

##########

metadata = pd.read_csv("../input/ashrae-energy-prediction/building_metadata.csv", dtype=metadata_dtype)

cols = list(metadata.columns[4:])

cols_imputed = metadata[cols].isnull().astype('uint8').add_suffix('_imputed')



imp = IterativeImputer(max_iter=10, verbose=0)

imp.fit(metadata.iloc[:,3:])

metadata_imputed = imp.transform(metadata.iloc[:,3:])

metadata_imputed = pd.concat([metadata.iloc[:,0:3],pd.DataFrame(metadata_imputed, columns=metadata.columns[3:]), cols_imputed], axis=1)

#metadata_imputed = pd.concat([metadata.iloc[:,0:3],pd.DataFrame(metadata_imputed, columns=metadata.columns[3:])], axis=1)

pd.DataFrame(metadata_imputed.isna().sum()/len(metadata_imputed),columns=["Metadata_Missing_Imputed"])



metadata_imputed.year_built = metadata_imputed.year_built.round()

metadata_imputed.floor_count = metadata_imputed.floor_count.round()



del metadata

del cols

del cols_imputed

gc.collect()



##########



test  = pd.merge(test,metadata_imputed,on='building_id',how='left')

print ("Testing Data+Metadata Shape {}".format(test.shape))

del metadata_imputed

gc.collect()



test  = pd.merge(test,weather_test_imputed,on=['site_id','timestamp_2'],how='left')

print ("Testing Data+Metadata+Weather Shape {}".format(test.shape))

del weather_test_imputed

gc.collect()



#

test.drop('timestamp_test',axis=1,inplace=True)

test.drop('timestamp_2',axis=1,inplace=True)

gc.collect()



#

test['number_unique_meter_per_building'] = test['building_id'].map(number_unique_meter_per_building)



test['mean_meter_reading_per_building'] = test['building_id'].map(mean_meter_reading_per_building)

test['median_meter_reading_per_building'] = test['building_id'].map(median_meter_reading_per_building)

test['std_meter_reading_per_building'] = test['building_id'].map(std_meter_reading_per_building)



test['mean_meter_reading_on_year_built'] = test['year_built'].map(mean_meter_reading_on_year_built)

test['median_meter_reading_on_year_built'] = test['year_built'].map(median_meter_reading_on_year_built)

test['std_meter_reading_on_year_built'] = test['year_built'].map(std_meter_reading_on_year_built)



test['mean_meter_reading_per_meter'] = test['meter'].map(mean_meter_reading_per_meter)

test['median_meter_reading_per_meter'] = test['meter'].map(median_meter_reading_per_meter)

test['std_meter_reading_per_meter'] = test['meter'].map(std_meter_reading_per_meter)



test['mean_meter_reading_per_primary_usage'] = test['primary_use'].map(mean_meter_reading_per_primary_usage)

test['median_meter_reading_per_primary_usage'] = test['primary_use'].map(median_meter_reading_per_primary_usage)

test['std_meter_reading_per_primary_usage'] = test['primary_use'].map(std_meter_reading_per_primary_usage)



test['mean_meter_reading_per_site_id'] = test['site_id'].map(mean_meter_reading_per_site_id)

test['median_meter_reading_per_site_id'] = test['site_id'].map(median_meter_reading_per_site_id)

test['std_meter_reading_per_site_id'] = test['site_id'].map(std_meter_reading_per_site_id)



#

for df in [test]:

    df['mean_meter_reading_per_building'] = df['mean_meter_reading_per_building'].astype("float16")

    df['median_meter_reading_per_building'] = df['mean_meter_reading_per_building'].astype("float16")

    df['std_meter_reading_per_building'] = df['std_meter_reading_per_building'].astype("float16")



    df['mean_meter_reading_on_year_built'] = df['mean_meter_reading_on_year_built'].astype("float16")

    df['median_meter_reading_on_year_built'] = df['median_meter_reading_on_year_built'].astype("float16")

    df['std_meter_reading_on_year_built'] = df['std_meter_reading_on_year_built'].astype("float16")



    df['mean_meter_reading_per_meter'] = df['mean_meter_reading_per_meter'].astype("float16")

    df['median_meter_reading_per_meter'] = df['median_meter_reading_per_meter'].astype("float16")

    df['std_meter_reading_per_meter'] = df['std_meter_reading_per_meter'].astype("float16")



    df['mean_meter_reading_per_primary_usage'] = df['mean_meter_reading_per_primary_usage'].astype("float16")

    df['median_meter_reading_per_primary_usage'] = df['median_meter_reading_per_primary_usage'].astype("float16")

    df['std_meter_reading_per_primary_usage'] = df['std_meter_reading_per_primary_usage'].astype("float16")



    df['mean_meter_reading_per_site_id'] = df['mean_meter_reading_per_site_id'].astype("float16")

    df['median_meter_reading_per_site_id'] = df['median_meter_reading_per_site_id'].astype("float16")

    df['std_meter_reading_per_site_id'] = df['std_meter_reading_per_site_id'].astype("float16")



    df['number_unique_meter_per_building'] = df['number_unique_meter_per_building'].astype('uint8')





#

le = LabelEncoder()

test['meter']= le.fit_transform(test['meter']).astype("uint8")

test['primary_use']= le.fit_transform(test['primary_use']).astype("uint8")

print(test.shape)



#

test.drop(to_drop,axis=1,inplace=True)

del to_drop

gc.collect()



%%time

predictions = []

step = 50000

for i in range(0, len(test), step):

    predictions.extend(np.expm1(reg.predict(test.iloc[i: min(i+step, len(test)), :], num_iteration=reg.best_iteration)))
%%time

Submission = pd.DataFrame(test.index,columns=['row_id'])

Submission['meter_reading'] = predictions

Submission['meter_reading'].clip(lower=0,upper=None,inplace=True)

Submission.to_csv("lgbm.csv",index=None)