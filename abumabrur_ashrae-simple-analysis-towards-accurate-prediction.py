

import numpy as np 

import pandas as pd 

from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split 

import xgboost as xgb

import plotly.express as px

from IPython.display import display

from datetime import datetime

import sklearn.metrics

import lightgbm as lgb

from sklearn.ensemble import RandomForestRegressor
train_df = pd.read_csv('/kaggle/input/ashrae-energy-prediction/train.csv')

weather_train_df = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_train.csv')
train_df.head()
train_df.dtypes
train_df['timestamp'] = train_df['timestamp'].astype('datetime64[ns]')
weather_train_df.head(3)
weather_train_df['timestamp'] = weather_train_df['timestamp'].astype('datetime64[ns]')
print('lengths of data in train and weather_train file are:{tr} and {we} respectively.'.format(tr = train_df.shape[0], we = weather_train_df.shape[0]))
test_df = pd.read_csv('/kaggle/input/ashrae-energy-prediction/test.csv')

weather_test_df = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_test.csv')

test_df['timestamp'] = test_df['timestamp'].astype('datetime64[ns]')

weather_test_df['timestamp'] = weather_test_df['timestamp'].astype('datetime64[ns]')
bld_data_df = pd.read_csv('/kaggle/input/ashrae-energy-prediction/building_metadata.csv')

bld_data_df.head(3)
bld_data_df.dtypes
train_data_df = train_df.copy()

train_data_df = train_data_df.merge(bld_data_df, on='building_id', how='left')

train_data_df = train_data_df.merge(weather_train_df, on=['site_id', 'timestamp'], how='left')
train_data_df.columns
train_data_df.head()
null_checks = pd.concat([train_data_df.isnull().sum(),train_data_df.isnull().sum()/train_data_df.isnull().count()*100],axis=1, keys = ['Total no of null entries','Percentage of null entries'])
null_checks.sort_values(by='Percentage of null entries', ascending = False)
mean_meter_reading =  train_data_df.groupby('timestamp')['meter_reading'].mean()
mean_meter_reading.plot(figsize=(14,8))
primary_use_agg_meter = train_data_df.groupby(['primary_use']).agg({'meter_reading':['count','sum', 'idxmax', 'max']})
primary_use_agg_meter
def reshape_agg_dataframe(agg_dataframe):

    level_0 = agg_dataframe.columns.droplevel(0)

    level_1 = agg_dataframe.columns.droplevel(1)

    level_0 = ['' if x == '' else '-' + x for x in level_0]

    agg_dataframe.columns = level_1 + level_0

    agg_dataframe.rename_axis(None, axis=1)

    return agg_dataframe
primary_use_agg_meter = reshape_agg_dataframe(primary_use_agg_meter)
primary_use_agg_meter.head(2)
primary_use_agg_meter.sort_values(by='meter_reading-sum', ascending= False)
train_data_df.iloc[8907488,:]
train_data_df['meter_type']= pd.Categorical(train_data_df['meter']).rename_categories({0: 'electricity', 1: 'chilledwater', 2: 'steam', 3: 'hotwater'})

daily_train = train_data_df.copy()

daily_train['date'] = daily_train['timestamp'].dt.date

del daily_train['meter']

daily_train = daily_train.groupby(['date', 'building_id', 'meter_type']).sum()

daily_train
daily_train_agg = daily_train.groupby(['date', 'meter_type']).agg({'meter_reading':['sum', 'mean', 'idxmax', 'max']})

daily_train_agg.head()
daily_train_agg = daily_train_agg.reset_index()

daily_train_agg = reshape_agg_dataframe(daily_train_agg)

daily_train_agg.head(3)
def show_figure(df,x_val, y_val,color_val, title_val):

    fig = px.line(df, x=x_val, y=y_val, color=color_val, render_mode='svg')

    fig.update_layout(title = title_val)

    fig.show(figsize=(16,12))

    return
show_figure(daily_train_agg,x_val='date',y_val='meter_reading-sum',color_val='meter_type',title_val='Total kWh per energy aspect')
daily_train_agg['building_id_max'] = [x[1] for x in daily_train_agg['meter_reading-idxmax']]

daily_train_agg.head()
print('Number of days that a building has the maximum electricity consumption of all the buildings:\n')

print(daily_train_agg[daily_train_agg['meter_type'] == 'electricity']['building_id_max'].value_counts())
daily_train_electricity = daily_train_agg[daily_train_agg['meter_type']=='electricity'].copy()

daily_train_electricity['building_id_max'] = pd.Categorical(daily_train_electricity['building_id_max'])

show_figure(daily_train_electricity,x_val='date',y_val='meter_reading-max',color_val='building_id_max',title_val='Maximum consumption values for the day and energy aspect')
print('Number of days that a building has the maximum chilledwater consumption of all the buildings:\n')

print(daily_train_agg[daily_train_agg['meter_type'] == 'chilledwater']['building_id_max'].value_counts())
daily_train_chilledwater = daily_train_agg[daily_train_agg['meter_type']=='chilledwater'].copy()

daily_train_chilledwater['building_id_max'] = pd.Categorical(daily_train_chilledwater['building_id_max'])

show_figure(daily_train_chilledwater,x_val='date',y_val='meter_reading-max',color_val='building_id_max',title_val='Maximum consumption values for the day and energy aspect')
print('Number of days that a building has the maximum steam consumption of all the buildings:\n')

print(daily_train_agg[daily_train_agg['meter_type'] == 'steam']['building_id_max'].value_counts())
daily_train_steam = daily_train_agg[daily_train_agg['meter_type']=='steam'].copy()

daily_train_steam['building_id_max'] = pd.Categorical(daily_train_steam['building_id_max'])

show_figure(daily_train_steam,x_val='date',y_val='meter_reading-max',color_val='building_id_max',title_val='Maximum consumption values for the day and energy aspect')
print('Number of days that a building has the maximum hotwater consumption of all the buildings:\n')

print(daily_train_agg[daily_train_agg['meter_type'] == 'hotwater']['building_id_max'].value_counts())
daily_train_hotwater = daily_train_agg[daily_train_agg['meter_type']=='hotwater'].copy()

daily_train_hotwater['building_id_max'] = pd.Categorical(daily_train_hotwater['building_id_max'])

show_figure(daily_train_hotwater,x_val='date',y_val='meter_reading-max',color_val='building_id_max',title_val='Maximum consumption values for the day and energy aspect')
show_figure(daily_train_hotwater,x_val='date',y_val='meter_reading-max',color_val='building_id_max',title_val='Maximum consumption values for the day and energy aspect')
train_data_df.groupby('building_id').building_id.unique()
train_data_df_m = train_data_df.copy()
train_data_df_m.count()
lbl_encoder = LabelEncoder()

lbl_encoder.fit(train_data_df_m['primary_use'])

train_data_df_m['primary_use'] = np.uint8(lbl_encoder.transform(train_data_df_m['primary_use']))
train_data_df_m.head(2)
#determine which fields to be dropped

 

for col in ['timestamp','building_id','site_id','meter','meter_type','floor_count','year_built','cloud_coverage','precip_depth_1_hr']:

    del train_data_df_m[col]

 
#train_data_df_m = train_data_df_m.dropna() # determine if you really want to drop nan or fill them by interpolation#

train_data_target = train_data_df_m.loc[:,['meter_reading']]

del train_data_df_m['meter_reading']
#70% for training and 30% for evaluation/testing

x_train, x_val, y_train, y_val = train_test_split(train_data_df_m,train_data_target, test_size =0.3)
