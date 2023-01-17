import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import gc

from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_log_error

import hyperopt

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from hyperopt import space_eval

import time

import math

from hyperopt.pyll.base import scope

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from xgboost import plot_importance

from tqdm import tqdm_notebook as tqdm

import catboost

import lightgbm as lgb

from catboost import CatBoostRegressor, Pool

from sklearn.preprocessing import LabelEncoder

from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

import pprint

pp = pprint.PrettyPrinter(indent=4)
project_dir = '/kaggle/input/ashrae-energy-prediction'

data_dir = project_dir
train = pd.read_csv(data_dir + "/train.csv")

build = pd.read_csv(data_dir + "/building_metadata.csv")

weather_train = pd.read_csv(data_dir + "/weather_train.csv")
def prepare_df(train, build, weather, should_compress=False, should_create_dummies=True):    

    train_build = train.merge(right=build,left_on="building_id", right_on="building_id", how="left")

    train_build = train_build.query('not (building_id <= 104 & meter == 0 & timestamp <= "2016-05-20")')

    train_build["month"] = pd.to_datetime(train_build["timestamp"]).dt.month.astype(np.int8)

    train_build["year"] = pd.to_datetime(train_build["timestamp"]).dt.year

    train_build["day_of_week"] = pd.to_datetime(train_build["timestamp"]).dt.dayofweek

    dates_range = pd.date_range(start='2015-12-31', end='2019-01-01')

    us_holidays = calendar().holidays(start=dates_range.min(), end=dates_range.max())

    train_build['is_holiday'] = (pd.to_datetime(train_build['timestamp']).dt.date.astype('datetime64').isin(us_holidays))

    train_build.loc[(train_build['day_of_week'] == 5) | (train_build['day_of_week'] == 6) , 'is_holiday'] = True

    train_build["weekofmonth"] = np.ceil(pd.to_datetime(train_build["timestamp"]).dt.day/7).astype(np.int8)

    month_season_bins = pd.IntervalIndex.from_tuples([(1, 2), (3, 5), (6, 8), (9, 11), (12, 12)])

    encoder = LabelEncoder()

    train_build["season"] =  encoder.fit_transform(pd.cut(train_build["month"], month_season_bins).astype(str)).astype(np.uint8)

    train_build["hour"] = pd.to_datetime(train_build["timestamp"]).dt.hour.astype(np.int8)

#     half = np.floor(pd.to_datetime(train_build["timestamp"]).dt.minute/30).astype(np.int8)

#     train_build["hour_half"] = (train_build["hour"] * 2 + half).astype(np.int8)

    train_build["is_working_hour"] = (train_build["hour"] >= 8) & (train_build["hour"] <= 18)

    train_build['square_feet'] = np.log(train_build['square_feet'])

    train_build["square_feet_per_floor"] = train_build["square_feet"]/train_build["floor_count"]

    train_build["square_feet_multiplied_by_floor"] = train_build["square_feet"] * train_build["floor_count"]

    train_build["age"] = train_build["year"] - train_build["year_built"]

    train_build.loc[(train_build['primary_use'] == "Education") & (train_build['month'] >= 6) & (train_build['month'] <= 8), 'is_vacation_month'] = np.int8(1)

    train_build.loc[train_build['is_vacation_month']!=1, 'is_vacation_month'] = np.int8(0)

    encoder = LabelEncoder()

    train_build["primary_use"] = encoder.fit_transform(train_build["primary_use"]).astype(np.uint8)

    

    train_build_weather = train_build.merge(right=weather, left_on=["timestamp", "site_id"], right_on=["timestamp", "site_id"], how="left")

    train_build_weather["air_temperature_2"] = train_build_weather["air_temperature"] ** 2

    train_build_weather["dew_temperature_2"] = train_build_weather["dew_temperature"] ** 2  

    train_build_weather["wind_direction_cat"] = train_build_weather["wind_direction"]

    replacement = {}

    for i in range(0, 351, 30):

        key = tuple(list(range(i, i+30)))

        replacement[key] = i

    for (key,value) in replacement.items():

        train_build_weather["wind_direction_cat"].replace(key, value, inplace=True)

        gc.collect()

    

    beaufort = [(0, 0, 0.3), (1, 0.3, 1.6), (2, 1.6, 3.4), (3, 3.4, 5.5), (4, 5.5, 8), (5, 8, 10.8), (6, 10.8, 13.9), 

          (7, 13.9, 17.2), (8, 17.2, 20.8), (9, 20.8, 24.5), (10, 24.5, 28.5), (11, 28.5, 33), (12, 33, 200)]

    

    for item in beaufort:

        train_build_weather.loc[(train_build_weather['wind_speed']>=item[1]) & (train_build_weather['wind_speed']<item[2]), 'beaufort_scale'] = item[0]

        gc.collect()

#     train_build_weather["is_weekend"] = pd.to_datetime(train_build_weather["timestamp"]).dt.dayofweek >= 5

    

    train_build_weather.drop(columns=["timestamp"], inplace=True)    

    if "meter_reading" in train_build_weather.columns:

      train_build_weather["meter_reading"] = np.log1p(train_build_weather['meter_reading']).astype(np.float32)

    return train_build_weather



cat_cols = [

            "hour",

            "is_working_hour",

            "is_holiday",

            "day_of_week",

            "season",

            "is_vacation_month",

            "site_id", 

            "building_id", 

            "meter",

            "primary_use",

           "wind_direction_cat"

           ]

cols_drop_x = [

               "weekofmonth", 

               "year",

               "year_built", 

               "meter_reading",

#                "wind_direction",

               "sea_level_pressure",

               "wind_speed",

                "wind_direction",

#     'precip_depth_1_hr'

              ]
def train_lgb(X_train, y_train, X_val, y_val, params):

    train_data = lgb.Dataset(X_train.values, 

                             label=y_train.values.ravel(),

                             feature_name=list(X_train.columns),

                             categorical_feature=cat_cols

                            )

    validation_data = lgb.Dataset(X_val.values, 

                             label=y_val.values.ravel(),

                             feature_name=list(X_val.columns),

                             categorical_feature=cat_cols

                            )

    evals_result = {}

    bst = lgb.train(params, train_data, 

                    valid_sets=[train_data, validation_data], 

                    valid_names=['train', 'val'], 

                    evals_result=evals_result, 

                    num_boost_round=10000,

                    early_stopping_rounds=100,

                    categorical_feature=cat_cols,

                   verbose_eval=100)

    return bst
gc.collect()

df = prepare_df(train, build, weather_train, should_compress=False, should_create_dummies=False).dropna(subset=["meter_reading"])

gc.collect()

# X_train = df.drop(columns=cols_drop_x)

# y_train = df[["meter_reading"]]



df_train_1 = df[df["month"] <= 6]

X_train_1 = df_train_1.drop(columns=cols_drop_x)

y_train_1 = df_train_1[["meter_reading"]]

df_train_1_sample = df_train_1.sample(n=100000, random_state=7)

del(df_train_1)

garbage = gc.collect()

X_train_1_sample = df_train_1_sample.drop(columns=cols_drop_x)

y_train_1_sample = df_train_1_sample[["meter_reading"]]

del(df_train_1_sample)

garbage = gc.collect()

df_train_2 = df[df["month"] > 6]

X_train_2 = df_train_2.drop(columns=cols_drop_x)

y_train_2 = df_train_2[["meter_reading"]]

df_train_2_sample = df_train_2.sample(n=100000, random_state=7)

del(df_train_2)

garbage = gc.collect()

X_train_2_sample = df_train_2_sample.drop(columns=cols_drop_x)

y_train_2_sample = df_train_2_sample[["meter_reading"]]

del(df_train_2_sample)

garbage = gc.collect()



# X_train_1 = df[df["weekofmonth"] == 1].drop(columns=cols_drop_x)

# y_train_1 = df[df["weekofmonth"] == 1][["meter_reading"]]

# # X_train_2 = df[df["weekofmonth"] == 2].drop(columns=cols_drop_x)

# # y_train_2 = df[df["weekofmonth"] == 2][["meter_reading"]]

# # X_train_3 = df[df["weekofmonth"] == 3].drop(columns=cols_drop_x)

# # y_train_3 = df[df["weekofmonth"] == 3][["meter_reading"]]

# X_train_4 = df[df["weekofmonth"] == 4].drop(columns=cols_drop_x)

# y_train_4 = df[df["weekofmonth"] == 4][["meter_reading"]]



del (df)

gc.collect()

models = []

best_params_for_max_depths = [

        #-1

   {   

    'bagging_fraction': 0.9,

    'bagging_freq': 1,

    'eval_metric': 'RMSE',

    'feature_fraction': 0.85,

#     'lambda_l1': 1,

    'lambda_l2': 2,

    'learning_rate': 0.05,

    'loss_function': 'RMSE',

    'max_bin': 250,

    'max_depth': 20,

    'metric': 'rmse',

#     'min_sum_hessian_in_leaf': 30,

    'num_leaves': 800,

    'objective': 'regression',

    'random_state': 42,

    'verbose': None},

#     {

#         "objective": "regression",

#     "boosting": "gbdt",#dart,gbdt

#     "num_leaves": 45,

#     "learning_rate": 0.02,

#     "feature_fraction": 0.9,

#     "reg_lambda": 2,

#     "metric": "rmse"

#     }

]



for best_params in best_params_for_max_depths: 

    print("Training model with params: ")

    

    pp.pprint(best_params)

    garbage = gc.collect()

    

#     bst = train_lgb(X_train, y_train, best_params)

#     garbage = gc.collect()    

#     models.append(bst)

    

    bst1 = train_lgb(X_train=X_train_1, y_train=y_train_1, X_val=X_train_2_sample, y_val=y_train_2_sample, params=best_params)

    garbage = gc.collect()    

    models.append(bst1)

    

    bst2 = train_lgb(X_train=X_train_2, y_train=y_train_2, X_val=X_train_1_sample, y_val=y_train_1_sample, params=best_params)

    garbage = gc.collect()    

    models.append(bst2)

    

#     bst3 = train_lgb(X_train_3, y_train_3, best_params)

#     garbage = gc.collect()    

#     models.append(bst3)

    

#     bst4 = train_lgb(X_train_4, y_train_4, best_params)

#     garbage = gc.collect()    

#     models.append(bst4)

    

    print("Done training for above params.")

    

del(X_train_1)

del(y_train_1)

del(X_train_2)

del(y_train_2)

del(X_train_1_sample)

del(y_train_1_sample)

del(X_train_2_sample)

del(y_train_2_sample)

# del(X_train_3)

# del(y_train_3)

# del(X_train_4)

# del(y_train_4)



# del(X_train)

# del(y_train)



garbage = gc.collect()
test = pd.read_csv(data_dir + "/test.csv")

weather_test = pd.read_csv(data_dir + "/weather_test.csv")
y_preds = []
for i in tqdm( range(math.ceil(test.shape[0]/50000))):

    gc.collect()

    X = prepare_df(train=test.iloc[i*50000:min((i+1)*50000, test.shape[0])].drop(columns=["row_id"]), build=build, weather=weather_test, should_compress=False, should_create_dummies=False)

    X = X.drop(columns=[col for col in cols_drop_x if col != "meter_reading"])

    preds = np.zeros(shape=(X.shape[0],))

    for bst in models:

        preds_temp = bst.predict(X)

        preds_temp[preds_temp < 0] = 0

        preds_temp = np.expm1(preds_temp)

        preds += preds_temp

    preds /= len(models)

    y_preds += list(preds)

    gc.collect()
del(weather_test)

del(build)

garbage=gc.collect()
submission_df = pd.DataFrame(data={"row_id": test["row_id"], "meter_reading": y_preds})
submission_df.to_csv("submission.csv", index = False)