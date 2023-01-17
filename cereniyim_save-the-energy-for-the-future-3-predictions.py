# for data manipulation

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# pandas options

pd.set_option('display.max_columns', 50)

pd.set_option('display.float_format', lambda x: '%.3f' % x)

pd.set_option('mode.use_inf_as_na', True)

pd.options.mode.chained_assignment = None



# for date manipulation

from datetime import datetime



# for visualization: matplotlib

from matplotlib import pyplot as plt

from IPython.core.pylabtools import figsize

%matplotlib inline

# to display visuals in the notebook



# for visualization: seaborn

import seaborn as sns

sns.set_context(font_scale=2)



# for data preprocessing

from sklearn.preprocessing import LabelEncoder

from itertools import combinations

from sklearn.model_selection import KFold



# for building the model and calculate RMSE

import lightgbm as lgb

from sklearn.metrics import mean_squared_error

from math import sqrt



# to cleanup memory usage

import gc



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
## Function to reduce the DF size and reduce test dataframe size

def reduce_memory_usage(df, verbose=True):

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
# load training data created in the second notebook into dataframes

X = pd.read_csv("/kaggle/input/save-the-energy-for-the-future-2-fe-lightgbm/X.csv")

y = pd.read_csv("/kaggle/input/save-the-energy-for-the-future-2-fe-lightgbm/y.csv", header=None)



reduce_memory_usage(X)

reduce_memory_usage(y)
# rename target as log_meter_reading

y.rename(columns = {0: "log_meter_reading"}, 

         inplace=True)
# create categorical features 

categorical_features = ['building_id', 'site_id', 'meter',

                        'primary_use', 'wind_compass_direction',

                        'day_of_week', 'hour','is_weekend', 'season']



# initial parameters of light gbm algorithm

initial_params = {"objective": "regression",

                  "boosting": "gbdt",

                  "num_leaves": 60,

                  "learning_rate": 0.05,

                  "feature_fraction": 0.85,

                  "reg_lambda": 2,

                  "metric": {'rmse'}

}
# cretae kfold object and empty model and evaluation lists

kf = KFold(n_splits=4, shuffle=False, random_state=42)



# save 4 model as a list

models = []



# dynamically split X and y with the k-fold split indexes

for train_index,valid_index in kf.split(X):

    X_train_kf = X.loc[train_index]

    y_train_kf = y.loc[train_index]

    

    X_valid_kf = X.loc[valid_index]

    y_valid_kf = y.loc[valid_index]

    

    d_train = lgb.Dataset(X_train_kf, 

                          label=y_train_kf,

                          categorical_feature=categorical_features, 

                          free_raw_data=False)

    

    d_valid = lgb.Dataset(X_valid_kf, 

                          label=y_valid_kf,

                          categorical_feature=categorical_features, 

                          free_raw_data=False)

    

    model = lgb.train(initial_params, 

                      train_set=d_train, 

                      num_boost_round=1000, 

                      valid_sets=[d_train, d_valid],

                      verbose_eval=250, 

                      early_stopping_rounds=500)

    

    models.append(model)

    

    del X_train_kf, y_train_kf, X_valid_kf, y_valid_kf, d_train, d_valid

    gc.collect()
X.columns
del X

del y

gc.collect()
# load building  data modify

building = pd.read_csv("/kaggle/input/ashrae-energy-prediction/building_metadata.csv")

# drop floor_count

building.drop(columns=["floor_count"], inplace=True)



# load weather_test modify

weather_test = pd.read_csv("/kaggle/input/ashrae-energy-prediction/weather_test.csv")

weather_test["timestamp"] = pd.to_datetime(weather_test["timestamp"],

                                            format='%Y-%m-%d %H:%M:%S')



# load test and modify

test = pd.read_csv("/kaggle/input/ashrae-energy-prediction/test.csv")

test["timestamp"] = pd.to_datetime(test["timestamp"],

                                   format='%Y-%m-%d %H:%M:%S')



reduce_memory_usage(building)

reduce_memory_usage(weather_test)

reduce_memory_usage(test)
# add building age column

current_year = datetime.now().year

building['building_age'] = current_year - building['year_built']

building.drop(columns=['year_built'], inplace=True)



# since NA values only present in building age fillna can be used

building.fillna(round(building.building_age.mean(),0),

                inplace=True)
# create label encoder object and transform the column

le = LabelEncoder()

le_primary_use = le.fit_transform(building.primary_use)



# add label encoded column to dataframe

building['primary_use'] = le_primary_use



del le, le_primary_use

gc.collect()
# check if any NA values left

building.isna().sum()
def convert_season(month):

    if (month <= 2) | (month == 12):

        return 0

    # as winter

    elif month <= 5:

        return 1

    # as spring

    elif month <= 8:

        return 2

    # as summer

    elif month <= 11:

        return 3

    # as fall
# add month, day of week, day of month, hour, season

weather_test['month'] = weather_test['timestamp'].dt.month.astype(np.int8)

weather_test['day_of_week'] = weather_test['timestamp'].dt.dayofweek.astype(np.int8)

weather_test['day_of_month']= weather_test['timestamp'].dt.day.astype(np.int8)

weather_test['hour'] = weather_test['timestamp'].dt.hour

weather_test['season'] = weather_test.month.apply(convert_season)



# add is weekend column

weather_test['is_weekend'] = weather_test.day_of_week.apply(lambda x: 1 if x>=5 else 0)
# reset index for fast update

weather_test = weather_test.set_index(

    ['site_id','day_of_month','month'])
# create dataframe of daily means per site id

air_temperature_filler = pd.DataFrame(weather_test

                                      .groupby(['site_id','day_of_month','month'])

                                      ['air_temperature'].mean(),

                                      columns=["air_temperature"])

# create dataframe of air_temperatures to fill

temporary_df = pd.DataFrame({'air_temperature' : weather_test.air_temperature})



# update NA air_temperature values

temporary_df.update(air_temperature_filler, overwrite=False)



# update in the weather train dataset

weather_test["air_temperature"] = temporary_df["air_temperature"]



del temporary_df, air_temperature_filler

gc.collect()
# create dataframe of daily means per site id

cloud_coverage_filler = pd.DataFrame(weather_test

                                     .groupby(['site_id','day_of_month','month'])

                                     ['cloud_coverage'].mean(),

                                     columns = ['cloud_coverage'])

cloud_coverage_filler.fillna(round(cloud_coverage_filler.cloud_coverage.mean(),0), 

                             inplace=True)



# create dataframe of cloud_coverages to fill

temporary_df = pd.DataFrame({'cloud_coverage' : weather_test.cloud_coverage})



# update NA cloud_coverage values

temporary_df.update(cloud_coverage_filler, overwrite=False)



# update in the weather train dataset

weather_test["cloud_coverage"] = temporary_df["cloud_coverage"]



del temporary_df, cloud_coverage_filler

gc.collect()
# create dataframe of daily means per site id

dew_temperature_filler = pd.DataFrame(weather_test

                                      .groupby(['site_id','day_of_month','month'])

                                      ['dew_temperature'].mean(),

                                      columns=["dew_temperature"])

# create dataframe of dew_temperatures to fill

temporary_df = pd.DataFrame({'dew_temperature' : weather_test.dew_temperature})



# update NA dew_temperature values

temporary_df.update(dew_temperature_filler, overwrite=False)



# update in the weather train dataset

weather_test["dew_temperature"] = temporary_df["dew_temperature"]



del temporary_df, dew_temperature_filler

gc.collect()
# create dataframe of daily means per site id

precip_depth_filler = pd.DataFrame(weather_test

                                   .groupby(['site_id','day_of_month','month'])

                                   ['precip_depth_1_hr'].mean(),

                                   columns=['precip_depth_1_hr'])

precip_depth_filler.fillna(round(precip_depth_filler['precip_depth_1_hr'].mean(),0)

                           , inplace=True)



# create dataframe of precip_depth_1_hr to fill

temporary_df = pd.DataFrame({'precip_depth_1_hr' : weather_test.precip_depth_1_hr})



# update NA precip_depth_1_hr values

temporary_df.update(precip_depth_filler, overwrite=False)



# update in the weather train dataset

weather_test["precip_depth_1_hr"] = temporary_df["precip_depth_1_hr"]



del precip_depth_filler, temporary_df

gc.collect()
# create dataframe of daily means per site id

sea_level_filler = pd.DataFrame(weather_test

                                .groupby(['site_id','day_of_month','month'])

                                ['sea_level_pressure'].mean(),

                                columns=['sea_level_pressure'])

mean_sea_level_pressure = round(

    sea_level_filler

    ['sea_level_pressure']

    .astype(float)

    .mean(),2)



sea_level_filler.fillna(mean_sea_level_pressure, inplace=True)



# create dataframe of sea_level_pressure to fill

temporary_df = pd.DataFrame({'sea_level_pressure' : weather_test.sea_level_pressure})



# update NA sea_level_pressure values

temporary_df.update(sea_level_filler, overwrite=False)



# update in the weather train dataset

weather_test["sea_level_pressure"] = temporary_df["sea_level_pressure"]



del sea_level_filler, temporary_df

gc.collect()
# create dataframe of daily means per site id

wind_direction_filler = pd.DataFrame(weather_test

                                     .groupby(['site_id','day_of_month','month'])

                                     ['wind_direction'].mean(),

                                     columns=['wind_direction'])

# create dataframe of wind_direction to fill

temporary_df = pd.DataFrame({'wind_direction' : weather_test.wind_direction})



# update NA wind_direction values

temporary_df.update(wind_direction_filler, overwrite=False)



# update in the weather train dataset

weather_test["wind_direction"] = temporary_df["wind_direction"]



del temporary_df, wind_direction_filler

gc.collect()
# create dataframe of daily means per site id

wind_speed_filler = pd.DataFrame(weather_test

                                 .groupby(['site_id','day_of_month','month'])

                                 ['wind_speed'].mean(),

                                 columns=['wind_speed'])

# create dataframe of wind_speed to fill

temporary_df = pd.DataFrame({'wind_speed' : weather_test.wind_speed})



# update NA wind_speed values

temporary_df.update(wind_speed_filler, overwrite=False)



# update in the weather train dataset

weather_test["wind_speed"] = temporary_df["wind_speed"]



del temporary_df, wind_speed_filler

gc.collect()
# check if NA values are left

weather_test.isna().sum()
weather_test = weather_test.reset_index()
def convert_direction(degrees):

    if degrees <= 90:

        return 0

    # as norteast direction

    elif degrees <= 180:

        return 1

    # as southeast direction

    elif degrees <= 270:

        return 2

    # as southwest direction

    elif degrees <= 360:

        return 3

    # as northwest direction

weather_test['wind_compass_direction'] = weather_test.wind_direction.apply(convert_direction)

weather_test.drop(columns=['wind_direction'], inplace=True)
# create weather variables combinations

weather_test['meansea_level_pressure_wind_speed'] = (weather_test['sea_level_pressure'] +

                                                     weather_test['wind_speed']) / 2

weather_test['meancloud_coverage_sea_level_pressure'] = (weather_test['sea_level_pressure'] + 

                                                         weather_test['cloud_coverage']) / 2

weather_test['meancloud_coverage_wind_speed '] = (weather_test['cloud_coverage'] + 

                                                  weather_test['wind_speed']) / 2

weather_test['meanprecip_depth_1_hr_sea_level_pressure'] = (weather_test['precip_depth_1_hr'] + 

                                                            weather_test['sea_level_pressure']) / 2

weather_test['meanair_temperature_sea_level_pressure'] = (weather_test['air_temperature'] + 

                                                          weather_test['sea_level_pressure']) / 2
# merge dataframes on test dataframe

test = test.merge(building, on = "building_id", how = "left")

test = test.merge(weather_test, on = ["site_id", "timestamp"], how="left")



# delete the other ones to save space from the memory

del weather_test

del building

gc.collect()
test.columns
test.drop(columns = ["row_id", 

                     "timestamp"], inplace=True)
#feature_set = ['building_age', 'le_primary_use', 'cloud_coverage',

#               'is_weekend','wind_speed', 'day_of_week',

#               'wind_compass_direction', 'sea_level_pressure', 'air_temperature',

#               'day_of_month', 'dew_temperature', 'hour', 

#               'month', 'meter', 'building_id', 

#               'site_id', 'floor_count', 'square_feet', 'year']
print("Number of unique columns in the test dataset:", test.shape[1])
test.isna().sum()
# split test set into two for faster imputations

X_test_2017 = test[:20848800]

X_test_2018 = test[20848800:]



del test

gc.collect()
X_test_2017 = X_test_2017.fillna(method='ffill', axis=1)

reduce_memory_usage(X_test_2017)

gc.collect()
X_test_2018 = X_test_2018.fillna(method='ffill', axis=1)

reduce_memory_usage(X_test_2018)

gc.collect()
print('2017 Test Data Shape:', X_test_2017.shape)
print('2018 Test Data Shape:', X_test_2018.shape)
X_test_2017.memory_usage()
X_test_2017.dtypes
# features that datatypes to be converted

int_features = ['building_age', 'primary_use', 

                'is_weekend',  'wind_compass_direction']



for feature in int_features:

    X_test_2017[feature] = X_test_2017[feature].astype('int8')

    X_test_2018[feature] = X_test_2018[feature].astype('int8')
X_test_2017.memory_usage()
gc.collect()
predictions_2017 = []



for model in models:

    if  predictions_2017 == []:

        predictions_2017 = (np

                            .expm1(model

                                   .predict(X_test_2017, 

                                            num_iteration=model.best_iteration)) / len(models))

    else:

        predictions_2017 += (np

                             .expm1(model

                                    .predict(X_test_2017,

                                             num_iteration=model.best_iteration)) / len(models))
del X_test_2017

gc.collect()
predictions_2018 = []



for model in models:

    if  predictions_2018 == []:

        predictions_2018 = (np

                            .expm1(model

                                   .predict(X_test_2018, 

                                            num_iteration=model.best_iteration)) / len(models))

    else:

        predictions_2018 += (np

                             .expm1(model

                                    .predict(X_test_2018, 

                                             num_iteration=model.best_iteration)) / len(models))
del X_test_2018

gc.collect()
for model in models:

    lgb.plot_importance(model)

    plt.show()
# to fetch row_ids

sample_submission = pd.read_csv("/kaggle/input/ashrae-energy-prediction/sample_submission.csv")

row_ids = sample_submission.row_id



del sample_submission

gc.collect()
# make sure of the shape of predictions

predictions_2017.shape
predictions_2018.shape
# split row_id's with the indexes of 2017 and 2018 predictions

row_ids_2017 = row_ids[:predictions_2017.shape[0]]

row_ids_2018 = row_ids[predictions_2018.shape[0]:]
submission_2017 = pd.DataFrame({"row_id": row_ids_2017, 

                                "meter_reading": np.clip(predictions_2017, 0, a_max=None)})



submission_2018 = pd.DataFrame({"row_id": row_ids_2018, 

                                "meter_reading": np.clip(predictions_2018, 0, a_max=None)})
submission = pd.concat([submission_2017,

                        submission_2018])



del submission_2017, submission_2018

gc.collect()
submission
submission.to_csv("submission.csv", index=False)
del models

gc.collect()