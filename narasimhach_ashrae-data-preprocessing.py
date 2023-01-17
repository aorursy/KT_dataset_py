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
import os

os.getcwd()
os.chdir("../input")
from sklearn.preprocessing import LabelEncoder

from sklearn import metrics
from sklearn.metrics import mean_squared_error

import lightgbm as lgb

from sklearn.model_selection import train_test_split
#########Reading all the files#########

train_df=pd.read_csv('/kaggle/input/ashrae-energy-prediction/train.csv')

test_df=pd.read_csv('/kaggle/input/ashrae-energy-prediction/test.csv')

weather_train_df=pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_train.csv')

weather_test_df=pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_test.csv')

building_meta_df=pd.read_csv('/kaggle/input/ashrae-energy-prediction/building_metadata.csv')
train_df.shape 

train_df.head()
######### Converting time stamp to datetime formats##########

train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])

test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])

weather_train_df['timestamp'] = pd.to_datetime(weather_train_df['timestamp'])

weather_test_df['timestamp'] = pd.to_datetime(weather_test_df['timestamp'])

    

building_meta_df['primary_use'] = building_meta_df['primary_use'].astype('category')
########### Merging data frames - step1 #################

temp_df = train_df[['building_id']]

temp_df = temp_df.merge(building_meta_df, on=['building_id'], how='left')

del temp_df['building_id']

train_df = pd.concat([train_df, temp_df], axis=1)



temp_df = test_df[['building_id']]

temp_df = temp_df.merge(building_meta_df, on=['building_id'], how='left')



del temp_df['building_id']

test_df = pd.concat([test_df, temp_df], axis=1)

del temp_df, building_meta_df
############Merging data frames step-2 ################

temp_df = train_df[['site_id','timestamp']]

temp_df = temp_df.merge(weather_train_df, on=['site_id','timestamp'], how='left')



del temp_df['site_id'], temp_df['timestamp']

train_df = pd.concat([train_df, temp_df], axis=1)



temp_df = test_df[['site_id','timestamp']]

temp_df = temp_df.merge(weather_test_df, on=['site_id','timestamp'], how='left')



del temp_df['site_id'], temp_df['timestamp']

test_df = pd.concat([test_df, temp_df], axis=1)



del temp_df, weather_train_df, weather_test_df
train_df.head()
test_df.head()
train_df.shape
train_df.describe()

#test_df.describe()
########To check the Missing values count by columns wise#################

print(train_df.isnull().sum())

print(test_df.isnull().sum())

###########Missing value imputation for both train and test data sets manually ###############

train_df['floor_count'] = train_df['floor_count'].fillna(-999).astype(np.int16)

test_df['floor_count'] = test_df['floor_count'].fillna(-999).astype(np.int16)

train_df['year_built'] = train_df['year_built'].fillna(-999).astype(np.int16)

test_df['year_built'] = test_df['year_built'].fillna(-999).astype(np.int16)

train_df['air_temperature']=train_df['air_temperature'].fillna(-999).astype(np.int16)

train_df['cloud_coverage']=train_df['cloud_coverage'].fillna(-999).astype(np.int16)

train_df['dew_temperature']=train_df['dew_temperature'].fillna(-999).astype(np.int16)

train_df['precip_depth_1_hr']=train_df['precip_depth_1_hr'].fillna(-999).astype(np.int16)

train_df['sea_level_pressure']=train_df['sea_level_pressure'].fillna(-999).astype(np.int16)

train_df['wind_direction']=train_df['wind_direction'].fillna(-999).astype(np.int16)

train_df['wind_speed']=train_df['wind_speed'].fillna(-999).astype(np.int16)



test_df['air_temperature']=test_df['air_temperature'].fillna(-999).astype(np.int16)

test_df['cloud_coverage']=test_df['cloud_coverage'].fillna(-999).astype(np.int16)

test_df['dew_temperature']=test_df['dew_temperature'].fillna(-999).astype(np.int16)

test_df['precip_depth_1_hr']=test_df['precip_depth_1_hr'].fillna(-999).astype(np.int16)

test_df['sea_level_pressure']=test_df['sea_level_pressure'].fillna(-999).astype(np.int16)

test_df['wind_direction']=test_df['wind_direction'].fillna(-999).astype(np.int16)

test_df['wind_speed']=test_df['wind_speed'].fillna(-999).astype(np.int16)
print(train_df.isnull().sum())