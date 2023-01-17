# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc



# matplotlib and seaborn for plotting

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



from sklearn.model_selection import train_test_split,KFold

from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb
%%time

root = '../input/ashrae-energy-prediction/'

train = pd.read_csv(root + 'train.csv', parse_dates=['timestamp'])

weather_train = pd.read_csv(root + 'weather_train.csv', parse_dates=['timestamp'])

test = pd.read_csv(root + 'test.csv', parse_dates=['timestamp'], usecols=['building_id','meter','timestamp'])

weather_test = pd.read_csv(root + 'weather_test.csv', parse_dates=['timestamp'])

metadata = pd.read_csv(root + 'building_metadata.csv')

sample_submission = pd.read_csv(root + 'sample_submission.csv')
print('Size of train data', train.shape)

print('Size of test data', train.shape)

print('Size of weather_train data', weather_train.shape)

print('Size of weather_test data', weather_test.shape)

print('Size of building_meta data', metadata.shape)
train['meter_reading'] = np.log1p(train['meter_reading'])
# Dropping floor_count variable as it has 75% missing values

metadata.drop('floor_count',axis=1,inplace=True)
metadata['primary_use'].unique()
metadata['primary_use'].replace({"Healthcare":"Other","Parking":"Other","Warehouse/storage":"Other","Manufacturing/industrial":"Other",

                                "Retail":"Other","Services":"Other","Technology/science":"Other","Food sales and service":"Other",

                                "Utility":"Other","Religious worship":"Other"},inplace=True)

metadata['square_feet'] = np.log1p(metadata['square_feet'])
np.mean(metadata['year_built'].astype(np.float32))
# metadata['year_built'].fillna(1968, inplace=True)

metadata['year_built'].fillna(-999, inplace=True)

metadata['year_built'] = metadata['year_built'].astype('int16')

# metadata['year_built'] = metadata['year_built'] - metadata['year_built'].min()
metadata.isna().sum()/len(metadata)
weather_train.isna().sum()/len(weather_train)
weather_train[weather_train['cloud_coverage'].isnull()]
weather_train.groupby(['site_id'])['cloud_coverage'].mean()
#Fill null values with mean value from each site

cols = ['air_temperature','cloud_coverage','dew_temperature','precip_depth_1_hr','sea_level_pressure','wind_direction','wind_speed']

weather_train = weather_train.set_index(['site_id'])

weather_test = weather_test.set_index(['site_id'])

for col in cols:

    weather_train[col].fillna(weather_train.groupby(['site_id'])[col].mean(),inplace=True)

    weather_test[col].fillna(weather_test[col].mean(),inplace=True)

    

weather_train = weather_train.reset_index()

weather_test = weather_test.reset_index()
weather_train[weather_train['cloud_coverage'].isnull()]
#As there are still null values (for sites where data is totally missing)

cols = ['air_temperature','cloud_coverage','dew_temperature','precip_depth_1_hr','sea_level_pressure','wind_direction','wind_speed']

for col in cols:

    weather_train[col].fillna(-999, inplace=True)

    weather_test[col].fillna(-999, inplace=True)
## Function to reduce the DF size, from: https://www.kaggle.com/caesarlupum/ashrae-start-here-a-gentle-introduction

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
train = reduce_mem_usage(train)

test = reduce_mem_usage(test)

weather_train = reduce_mem_usage(weather_train)

weather_test = reduce_mem_usage(weather_test)

metadata = reduce_mem_usage(metadata)
%%time

train = pd.merge(train,metadata,on='building_id',how='left')

test  = pd.merge(test,metadata,on='building_id',how='left')

print ("Training Data+Metadata Shape {}".format(train.shape))

print ("Testing Data+Metadata Shape {}".format(test.shape))

gc.collect()

train = pd.merge(train,weather_train,on=['site_id','timestamp'],how='left')

test  = pd.merge(test,weather_test,on=['site_id','timestamp'],how='left')

print ("Training Data+Metadata+Weather Shape {}".format(train.shape))

print ("Testing Data+Metadata+Weather Shape {}".format(test.shape))

gc.collect()
for df in [train, test]:

    df['Month'] = df['timestamp'].dt.month.astype("uint8")

    df['DayOfMonth'] = df['timestamp'].dt.day.astype("uint8")

    df['DayOfWeek'] = df['timestamp'].dt.dayofweek.astype("uint8")

    df['Hour'] = df['timestamp'].dt.hour.astype("uint8")
# Drop nonsense entries

# As per the discussion in the following thread, https://www.kaggle.com/c/ashrae-energy-prediction/discussion/117083, there is some discrepancy in the meter_readings for different ste_id's and buildings. It makes sense to delete them

idx_to_drop = list((train[(train['site_id'] == 0) & (train['timestamp'] < "2016-05-21 00:00:00")]).index)

print (len(idx_to_drop))

train.drop(idx_to_drop,axis='rows',inplace=True)



# dropping all the electricity meter readings that are 0, after considering them as anomalies.

idx_to_drop = list(train[(train['meter'] == 0) & (train['meter_reading'] == 0)].index)

print(len(idx_to_drop))

train.drop(idx_to_drop,axis='rows',inplace=True)
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
%%time

for df in [train, test]:

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
train.drop('timestamp',axis=1,inplace=True)

test.drop('timestamp',axis=1,inplace=True)
le = LabelEncoder()

train['primary_use'] = le.fit_transform(train['primary_use']).astype("uint8")

test['primary_use'] = le.fit_transform(test['primary_use']).astype("uint8")
%%time

# Let's check the correlation between the variables and eliminate the one's that have high correlation

# Threshold for removing correlated variables

threshold = 0.9



# Absolute value correlation matrix

corr_matrix = train.corr().abs()

# Upper triangle of correlations

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))



# Select columns with correlations above threshold

to_drop = [column for column in upper.columns if any(upper[column] > threshold)]



print('There are %d columns to remove.' % (len(to_drop)))

print ("Following columns can be dropped {}".format(to_drop))



train.drop(to_drop,axis=1,inplace=True)

test.drop(to_drop,axis=1,inplace=True)
train.head()
test.head()
y = train['meter_reading']

train.drop('meter_reading',axis=1,inplace=True)

categorical_cols = ['building_id','Month','meter','Hour','primary_use','DayOfWeek','DayOfMonth']
x_train,x_test,y_train,y_test = train_test_split(train,y,test_size=0.2,random_state=42)

print (x_train.shape)

print (y_train.shape)

print (x_test.shape)

print (y_test.shape)
test.shape
lgb_train = lgb.Dataset(x_train, y_train, categorical_feature=categorical_cols)

lgb_test = lgb.Dataset(x_test, y_test, categorical_feature=categorical_cols)

del x_train, x_test , y_train, y_test



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
del lgb_train,lgb_test

ser = pd.DataFrame(reg.feature_importance(),train.columns,columns=['Importance']).sort_values(by='Importance')

ser['Importance'].plot(kind='bar',figsize=(10,6))
del train
test.shape
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
#use onehotencoding instead of categorical_feature ? https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html