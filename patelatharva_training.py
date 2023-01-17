## This Python 3 environment comes with many helpful analytics libraries installed

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

import pprint

pp = pprint.PrettyPrinter(indent=4)

from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

# Any results you write to the current directory are saved as output.
# project_dir="/content/drive/My Drive/ashrae-energy-prediction"

# project_dir="."

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

#     train_build["is_weekend"] = train_build["day_of_week"] >= 5

    dates_range = pd.date_range(start='2015-12-31', end='2019-01-01')

    us_holidays = calendar().holidays(start=dates_range.min(), end=dates_range.max())

    train_build['is_holiday'] = (pd.to_datetime(train_build['timestamp']).dt.date.astype('datetime64').isin(us_holidays))

    train_build.loc[(train_build['day_of_week'] == 5) | (train_build['day_of_week'] == 6) , 'is_holiday'] = True

    train_build["weekofmonth"] = np.ceil(pd.to_datetime(train_build["timestamp"]).dt.day/7).astype(np.int8)

    month_season_bins = pd.IntervalIndex.from_tuples([(1, 2), (3, 5), (6, 8), (9, 11), (12, 12)])

    encoder = LabelEncoder()

    train_build["season"] =  encoder.fit_transform(pd.cut(train_build["month"], month_season_bins).astype(str)).astype(np.uint8)

    train_build["hour"] = pd.to_datetime(train_build["timestamp"]).dt.hour.astype(np.int8)

    half = np.floor(pd.to_datetime(train_build["timestamp"]).dt.minute/30).astype(np.int8)

    train_build["hour_half"] = (train_build["hour"] * 2 + half).astype(np.int8)

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

            "hour_half",

            "is_working_hour",

            "is_holiday",

#             "day_of_week",

            "season",

            "is_vacation_month",

            "site_id", 

            "building_id", 

            "meter",

            "primary_use",

           "wind_direction_cat"

           ]

cols_drop_x = [

                "hour",

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
number_of_evals = 200

print("number_of_evals", number_of_evals)

def find_best_params_for_lgb(X_train, y_train, X_val, y_val):

    evaluated_point_scores = {}

    

    def objective(params):

        garbage=gc.collect()

        if (str(params) in evaluated_point_scores):

            return evaluated_point_scores[str(params)]

        else:          

            

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

                     verbose_eval=100,

                   categorical_feature=cat_cols)

            

            y_val_preds = bst.predict(X_val)

            y_val_preds[y_val_preds < 0] = 0

            score = np.sqrt(mean_squared_log_error( np.expm1(y_val_preds), np.expm1(y_val) ))

#             print("Evaluating params:")

            pp.pprint(params)

#             print("rmsle: " + str(score))

            evaluated_point_scores[str(params)] = score

            f = open("params_lgb_score_"+ str(number_of_evals)+ ".json", "w+")

            f.write(str(evaluated_point_scores))

            f.close()

            return score



    param_space = {

        'objective': hp.choice("objective", ["regression"]),        

        "max_depth": scope.int(hp.quniform("max_depth", 6, 30, 1)),

        "learning_rate": hp.choice("learning_rate", [0.2]),

        "num_leaves": scope.int(hp.quniform("num_leaves", 50, 1000, 50)),

        "max_bin": scope.int(hp.quniform("max_bin", 50, 500, 10)),

        "bagging_fraction": hp.quniform('bagging_fraction', 0.50, 1.0, 0.05),

        "bagging_freq": hp.choice("bagging_freq", [1]),

        "loss_function": hp.choice("loss_function", ["RMSE"]), 

        "eval_metric": hp.choice("eval_metric", ["RMSE"]),

        "feature_fraction": hp.uniform("feature_fraction", 0.80, 1.0),

        "metric": hp.choice("metric", ["rmse"]),        

        "lambda_l1": hp.quniform('lambda_l1', 1.0, 10.0, 1),

        "lambda_l2": hp.quniform('lambda_l2', 1.0, 100.0, 5.0),

        "random_state": hp.choice("random_state", [7]),

        "verbose": hp.choice("verbose", [0]),

        "verbose_eval": hp.choice("verbose_eval", [False])

    }

    start_time = time.time()

    best_params = space_eval(

        param_space, 

        fmin(objective, 

             param_space, 

             algo=hyperopt.tpe.suggest,

             max_evals=number_of_evals))

    

    pp.pprint(best_params)

    elapsed_time = (time.time() - start_time) / 60

    print('Elapsed computation time: {:.3f} mins'.format(elapsed_time))

    print("Finding best number of iterations with learning rate: ", best_params["learning_rate"])



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

    bst = lgb.train(best_params, train_data, 

                    valid_sets=[train_data, validation_data], 

                    valid_names=['train', 'val'], 

                    evals_result=evals_result, 

                    num_boost_round=10000,

                    early_stopping_rounds=100,

                    verbose_eval=False,

                    categorical_feature=cat_cols)

    best_params["num_iterations"] = bst.best_iteration

    print ("Best params:")

    pp.pprint(best_params)

    return best_params
gc.collect()

df = prepare_df(

#                 train,

                train.sample(n=1000000, random_state=7), 

                build, weather_train, should_compress=False).dropna(subset=["meter_reading"])

garbage=gc.collect()

train_df = df[df["month"] <= 6]

val_df = df[df["month"] > 6].sample(n=100000, random_state=42)

print("training rows: ", train_df.shape[0])

print("validation rows: ", val_df.shape[0])

del(df)

X_train = train_df.drop(columns=cols_drop_x)

y_train = train_df[["meter_reading"]]

del(train_df)

X_val = val_df.drop(columns=cols_drop_x)

y_val = val_df[["meter_reading"]]

del(val_df)

garbage=gc.collect()
best_params = find_best_params_for_lgb(X_train, y_train, X_val, y_val)
f = open("best_params_lgb_"+str(number_of_evals)+"_.json", "w+")

f.write(str(best_params))

f.close()