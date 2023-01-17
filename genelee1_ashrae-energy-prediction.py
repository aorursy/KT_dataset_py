# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime

import matplotlib.pyplot as plt

import gc

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold

import lightgbm as lgb

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
bm_data = pd.read_csv('/kaggle/input/ashrae-energy-prediction/building_metadata.csv')

bm_data.head()
w_train_data = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_train.csv')

w_train_data.head()

w_train_data.shape
train_data = pd.read_csv('/kaggle/input/ashrae-energy-prediction/train.csv')



# Removing weird data on site_id = 0 before '2016-05-20'

train_data = train_data [ train_data['building_id'] != 1099 ]

train_data = train_data.query('not(building_id <= 104 & meter == 0 & timestamp <= "2016-05-20")')

w_test_data = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_test.csv')

w_test_data.head()
test_data = pd.read_csv('/kaggle/input/ashrae-energy-prediction/test.csv')

test_data.head()
# Function for viewing missing values

def missing_values(df):

    mv = pd.DataFrame(df.isnull().sum()).reset_index()

    mv.columns = ['Column Name', 'Missing Values']

    mv['Total Rows'] = df.shape[0]

    mv['Missing Percentage'] = round((mv['Missing Values']/mv['Total Rows'])*100, 2)

    return mv
def fill_weather_data(weather_df):

    # Handling Missing Hours. The data has 16 sites in 2016. So it should have (16*24*366 = 140544) records instead of 139773 records. Hence, 771 records of hours are missing

    time_format = '%Y-%m-%d %H:%M:%S'

    start_date = datetime.datetime.strptime(weather_df['timestamp'].min(), time_format)

    end_date = datetime.datetime.strptime(weather_df['timestamp'].max(), time_format)

    total_hours = int(((end_date - start_date).total_seconds() + 3600) / 3600)

    hours_list = [(end_date - datetime.timedelta(hours = x)).strftime(time_format) for x in range(total_hours)]



    missing_hours = []

    for site_id in range(16):

        site_hours = np.array(weather_df[weather_df['site_id'] == site_id]['timestamp'])

        new_rows = pd.DataFrame(np.setdiff1d(hours_list, site_hours), columns = ['timestamp'])

        new_rows['site_id'] = site_id

        weather_df = pd.concat([weather_df, new_rows])



        weather_df = weather_df.reset_index(drop = True)

    

    # Adding Day, Week, Month columns

    weather_df['datetime'] = pd.to_datetime(weather_df['timestamp'])

    weather_df['day'] = weather_df['datetime'].dt.day

    weather_df['week'] = weather_df['datetime'].dt.week

    weather_df['month'] = weather_df['datetime'].dt.month

    weather_df = weather_df.set_index(['site_id', 'day', 'month'])

    

    # Filling missing air_temperature values by mean of each month

    fill_air_temperature = pd.DataFrame(weather_df.groupby(['site_id', 'day', 'month'])['air_temperature'].mean(), columns=['air_temperature'])

    weather_df.update(fill_air_temperature, overwrite = False)

    

    # Fill missing cloud_coverage values by first calculating mean of each month then propogating missing last observation to missing values

    fill_cloud_coverage = weather_df.groupby(['site_id', 'day', 'month'])['cloud_coverage'].mean()

    fill_cloud_coverage = pd.DataFrame(fill_cloud_coverage.fillna(method='ffill'), columns=['cloud_coverage'])

    weather_df.update(fill_cloud_coverage, overwrite = False)

    

    # Fill missing dew_temperature values with monthly mean

    fill_dew_temperature = pd.DataFrame(weather_df.groupby(['site_id', 'day', 'month'])['dew_temperature'].mean(), columns =['dew_temperature'])

    weather_df.update(fill_dew_temperature, overwrite = False)



    # Fill missing precip_depth_1_hr values by propagating monthly average to missing values

    fill_precip_depth = weather_df.groupby(['site_id', 'day', 'month'])['precip_depth_1_hr'].mean()

    fill_precip_depth = pd.DataFrame(fill_precip_depth.fillna(method = 'ffill'), columns = ['precip_depth_1_hr'])

    weather_df.update(fill_precip_depth, overwrite = False)

    

    # Fill missing sea level values by propagating monthly mean to missing values

    fill_sea_level_pressure = weather_df.groupby(['site_id', 'day', 'month'])['sea_level_pressure'].mean()

    fill_sea_level_pressure = pd.DataFrame(fill_sea_level_pressure.fillna(method = 'ffill'), columns = ['sea_level_pressure'])

    weather_df.update(fill_sea_level_pressure, overwrite = False)



    # Fill missing wind_speed values with monthly average

    fill_wind_speed = pd.DataFrame(weather_df.groupby(['site_id', 'day', 'month'])['wind_speed'].mean(), columns = ['wind_speed'])

    weather_df.update(fill_wind_speed, overwrite = False)

    

    # Drop unneccassary columns

    weather_df = weather_df.reset_index() 

    weather_df = weather_df.drop(['wind_direction', 'datetime', 'day', 'week', 'month'], axis = 1)

    

    return weather_df
w_train_data = fill_weather_data(w_train_data)
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
train_data = reduce_mem_usage(train_data, use_float16 = True)

bm_data = reduce_mem_usage(bm_data, use_float16 = True)

w_train_data = reduce_mem_usage(w_train_data, use_float16 = True)
# Merging train data with weather data

train_data = train_data.merge(bm_data, left_on = 'building_id', right_on = 'building_id', how = 'left') 

train_data = train_data.merge(w_train_data, left_on = ['site_id', 'timestamp'], right_on = ['site_id', 'timestamp'], how = 'left')

del w_train_data

gc.collect()
train_data.sort_values('timestamp')

train_data.reset_index(drop = True)



train_data['timestamp'] = pd.to_datetime(train_data['timestamp'], format = '%Y-%m-%d %H:%M:%S')

train_data['hour'] = train_data['timestamp'].dt.hour

train_data['weekend'] = train_data['timestamp'].dt.weekday

train_data['square_feet'] = np.log1p(train_data['square_feet'])

train_data = train_data.drop(['timestamp', 'sea_level_pressure', 'wind_speed', 'year_built', 'floor_count'], axis = 1)

gc.collect()

le = LabelEncoder()

train_data['primary_use'] = le.fit_transform(train_data['primary_use'])
target = np.log1p(train_data['meter_reading'])

features = train_data.drop(['meter_reading'], axis = 1)

del train_data

gc.collect()
categorical_features = ["building_id", "site_id", "meter", "primary_use", "weekend"]

params = {

    "objective": "regression",

    "boosting": "gbdt",

    "num_leaves": 1280,

    "learning_rate": 0.05,

    "feature_fraction": 0.85,

    "reg_lambda": 2,

    "metric": "rmse",

}



kf = KFold(n_splits=3)

models = []

for train_index,test_index in kf.split(features):

    train_features = features.loc[train_index]

    train_target = target.loc[train_index]

    

    test_features = features.loc[test_index]

    test_target = target.loc[test_index]

    

    d_training = lgb.Dataset(train_features, label=train_target,categorical_feature=categorical_features, free_raw_data=False)

    d_test = lgb.Dataset(test_features, label=test_target,categorical_feature=categorical_features, free_raw_data=False)

    

    model = lgb.train(params, train_set=d_training, num_boost_round=300, valid_sets=[d_training,d_test], verbose_eval=25, early_stopping_rounds=50)

    models.append(model)

    del train_features, train_target, test_features, test_target, d_training, d_test

    gc.collect()
del features, target

gc.collect()
for model in models:

    lgb.plot_importance(model)

    plt.show()
row_ids = test_data['row_id']

test_data.drop(['row_id'], axis = 1)

test_data = reduce_mem_usage(test_data)
test_data = test_data.merge(bm_data, left_on = 'building_id', right_on='building_id',

                           how = 'left')

del bm_data

gc.collect()
w_test_data = fill_weather_data(w_test_data)

w_test_data = reduce_mem_usage(w_test_data)
test_data = test_data.merge(w_test_data, how = 'left', on = ['timestamp','site_id'])

del w_test_data

gc.collect()
test_data.sort_values('timestamp')

test_data.reset_index(drop = True)



test_data['timestamp'] = pd.to_datetime(test_data['timestamp'], format = '%Y-%m-%d %H:%M:%S')

test_data['hour'] = test_data['timestamp'].dt.hour

test_data['weekend'] = test_data['timestamp'].dt.weekday

test_data['square_feet'] = np.log1p(test_data['square_feet'])

test_data = test_data.drop(['row_id', 'timestamp', 'sea_level_pressure', 'wind_speed', 'year_built', 'floor_count'], axis = 1)

gc.collect()

le = LabelEncoder()

test_data['primary_use'] = le.fit_transform(test_data['primary_use'])
# Prediction

results = []

for model in models:

    if results == []:

        results = np.expm1(model.predict(test_data, num_iteration=model.best_iteration)) / len(models)

    else:

        results += np.expm1(model.predict(test_data, num_iteration = model.best_iteration)) / len(models)

    del model

    gc.collect()
del test_data, models

gc.collect()

results_data = pd.DataFrame({'row_id': row_ids, 'meter_reading': np.clip(results, 0, a_max=None)})

del row_ids, results

gc.collect()

results_data.to_csv('submission.csv', index = False)