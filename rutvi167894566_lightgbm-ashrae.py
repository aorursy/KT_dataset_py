import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from plotly import tools

#import plotly.plotly as py

import plotly.figure_factory as ff

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

init_notebook_mode(connected=True)
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
train = pd.read_csv('../input/ashrae-energy-prediction/train.csv')

weather_train = pd.read_csv('../input/ashrae-energy-prediction/weather_train.csv')

test = pd.read_csv('../input/ashrae-energy-prediction/test.csv')

weather_test = pd.read_csv('../input/ashrae-energy-prediction/weather_test.csv')

building_metadata = pd.read_csv('../input/ashrae-energy-prediction/building_metadata.csv')

#sample_submission = pd.read_csv('sample_submission.csv')
building_metadata.head()
weather_train.head(3)
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
##########Reducing memory########################

train = reduce_mem_usage(train)

test = reduce_mem_usage(test)

weather_train = reduce_mem_usage(weather_train)

weather_test = reduce_mem_usage(weather_test)

building_meta = reduce_mem_usage(building_metadata)

train.head(5)
train_df = train.merge(building_metadata, on='building_id', how='left')

train = train_df.merge(weather_train, on=['site_id', 'timestamp'], how='left')

train.head(5)

test_df = test.merge(building_metadata, on='building_id', how='left')

test = test_df.merge(weather_test, on=['site_id', 'timestamp'], how='left')

test.head(5)
train_df.head(3)
train.drop('timestamp',axis=1,inplace=True)

test.drop('timestamp',axis=1,inplace=True)
train.head()
columns = ['air_temperature','cloud_coverage','dew_temperature','precip_depth_1_hr','sea_level_pressure','wind_direction','wind_speed']

train.loc[:, columns] = train.loc[:, columns].interpolate(method ='linear', limit_direction ='forward') 

test.loc[:, columns] = test.loc[:, columns].interpolate(method ='linear', limit_direction ='forward') 
train.head()
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

le = LabelEncoder()



train['meter']= le.fit_transform(train['meter']).astype("uint8")

test['meter']= le.fit_transform(test['meter']).astype("uint8")

train['primary_use']= le.fit_transform(train['primary_use']).astype("uint8")

test['primary_use']= le.fit_transform(test['primary_use']).astype("uint8")
threshold = 0.9
correlation = train.corr().abs()

correlation.head()
correlation.head()
test_1 = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(np.bool))

test_1.head()
threshold=0.9
to_drop = [column for column in test_1.columns if any(test_1[column] > threshold)]
train.head()
#train.drop(to_drop,axis=1,inplace=True)

test.drop(to_drop,axis=1,inplace=True)

y = train['meter_reading']

train.drop('meter_reading',axis=1,inplace=True)

train.drop('site_id',axis=1,inplace=True)
test.head()
! pip install lightgbm
cat_cols = ['building_id', 'primary_use','year_built', 'meter',  'wind_direction']
from sklearn.model_selection import train_test_split,KFold

import lightgbm as lgb

x_train,x_test,y_train,y_test = train_test_split(train,y,test_size=0.25,random_state=42)

print (x_train.shape)

print (y_train.shape)

print (x_test.shape)

print (y_test.shape)



lgb_train = lgb.Dataset(x_train, y_train ,categorical_feature=cat_cols)

lgb_test = lgb.Dataset(x_test, y_test ,categorical_feature=cat_cols)

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

          'random_state': 47,

          "num_leaves": 41}
reg = lgb.train(params, lgb_train, num_boost_round=3000, valid_sets=[lgb_train, lgb_test], early_stopping_rounds=100, verbose_eval = 100)
test.drop('row_id',axis=1,inplace=True)
#del lgb_train,lgb_test
Submission_file = pd.DataFrame(test.index,columns=['row_id'])
Submission_file
prediction = []

step = 100000

for i in range(0, len(test), step):

    prediction.extend(np.expm1(reg.predict(test.iloc[i: min(i+step, len(test)), :], num_iteration=reg.best_iteration)))

Submission_file['meter_reading'] = prediction

Submission_file['meter_reading'].clip(lower=0,upper=None,inplace=True)

Submission_file.to_csv("Twentysix.csv",index=None)