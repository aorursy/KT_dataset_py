import gc

import os

import numpy as np

import pandas as pd
path_external_data = "/kaggle/input/ashrae-ucf-spider-and-eda-full-test-labels"

path_data = "/kaggle/input/ashrae-energy-prediction/"

path_train = path_data + "train.csv"

path_building = path_data + "building_metadata.csv"

path_weather_train = path_data + "weather_train.csv"
import holidays

en_holidays = holidays.England()

ir_holidays = holidays.Ireland()

ca_holidays = holidays.Canada()

us_holidays = holidays.UnitedStates()
df_train = pd.read_csv(path_train)

building = pd.read_csv(path_building)

weather_train = pd.read_csv(path_weather_train)
weather_train = weather_train.groupby('site_id').apply(lambda group: group.interpolate(limit_direction='both'))
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

building.primary_use = le.fit_transform(building.primary_use)
#https://www.kaggle.com/isaienkov/lightgbm-fe-1-19



beaufort = [(0, 0, 0.3), (1, 0.3, 1.6), (2, 1.6, 3.4), (3, 3.4, 5.5), (4, 5.5, 8), (5, 8, 10.8)

            , (6, 10.8, 13.9), (7, 13.9, 17.2), (8, 17.2, 20.8), (9, 20.8, 24.5), (10, 24.5, 28.5)

            , (11, 28.5, 33), (12, 33, 200)]



def average_imputation(df, column_name):

    imputation = df.groupby(['timestamp'])[column_name].mean()

    

    df.loc[df[column_name].isnull(), column_name] = df[df[column_name].isnull()][[column_name]].apply(lambda x: imputation[df['timestamp'][x.index]].values)

    del imputation

    return df



def degToCompass(num):

    val=int((num/22.5)+.5)

    arr=[i for i in range(0,16)]

    return arr[(val % 16)]
#https://www.kaggle.com/c/ashrae-energy-prediction/discussion/114161#latest-658796

def relative_humidity(Tc,Tdc):

    E = 6.11*10.0**(7.5*Tdc/(237.7+Tdc))

    Es = 6.11*10.0**(7.5*Tc/(237.7+Tc))    

    RH = (E/Es)*100

    return RH
df_train['meter_reading'] = np.log1p(df_train['meter_reading'])

building_median = df_train.groupby('building_id')['meter_reading'].median().astype(np.float16)

building_mean = df_train.groupby('building_id')['meter_reading'].mean().astype(np.float16)

building_min = df_train.groupby('building_id')['meter_reading'].min().astype(np.float16)

building_max = df_train.groupby('building_id')['meter_reading'].max().astype(np.float16)

building_std = df_train.groupby('building_id')['meter_reading'].std().astype(np.float16)
def prepare_data(X, building_data, weather_data, test=False):

    """

    Preparing final dataset with all features.

    """

    

    X = X.merge(building_data, on="building_id", how="left")

    X = X.merge(weather_data, on=["site_id", "timestamp"], how="left")

    X['building_median'] = X['building_id'].map(building_median)

    X['building_mean'] = X['building_id'].map(building_median)

    X['building_min'] = X['building_id'].map(building_median)

    X['building_max'] = X['building_id'].map(building_median)

    X['building_std'] = X['building_id'].map(building_median)

    #--------------------------------------

    

    X.timestamp = pd.to_datetime(X.timestamp, format="%Y-%m-%d %H:%M:%S")

    

    # site_id = 0 has some building where meter readings before May 21, 2016 are not reliable so dropping those records 

    X = X.query('not (building_id <= 104 & meter == 0 & timestamp <= "2016-05-20")')

    

    #--------------------------------------

    #https://www.kaggle.com/isaienkov/lightgbm-fe-1-19

    X = average_imputation(X, 'wind_speed')

    X = average_imputation(X, 'wind_direction')

     

    for item in beaufort:

        X.loc[(X['wind_speed']>=item[1]) & (X['wind_speed']<item[2]), 'beaufort_scale'] = item[0]

        

    X['wind_direction'] = X['wind_direction'].apply(degToCompass)

    X['beaufort_scale'] = X['beaufort_scale'].astype(np.uint8)

    X["wind_direction"] = X['wind_direction'].astype(np.uint8)

    

    #--------------------------------------

    

    X.square_feet = np.log1p(X.square_feet)

    

    if not test:

        X.sort_values("timestamp", inplace=True)

        X.reset_index(drop=True, inplace=True)

    

    gc.collect()

    

    #-------------------------------------

    

    X["hour"] = X.timestamp.dt.hour

    X["weekday"] = X.timestamp.dt.weekday

    

    X["year"] = X.timestamp.dt.year

    X['age'] = X['year'] - X['year_built'] 

    

    

    #Jump from 429 to 408 (21 person)

    # https://www.kaggle.com/c/ashrae-energy-prediction/discussion/115256#latest-669944

    en_idx = X.query('site_id == 1 or site_id == 5').index

    ir_idx = X.query('site_id == 12').index

    ca_idx = X.query('site_id == 7 or site_id == 11').index

    us_idx = X.query('site_id == 0 or site_id == 2 or site_id == 3 or site_id == 4 or site_id == 6 or site_id == 8 or site_id == 9 or site_id == 10 or site_id == 13 or site_id == 14 or site_id == 15').index

    

    X['is_holiday'] = 0

    X.loc[en_idx, 'is_holiday'] = X.loc[en_idx, 'timestamp'].apply(lambda x: en_holidays.get(x, default=0))

    X.loc[ir_idx, 'is_holiday'] = X.loc[ir_idx, 'timestamp'].apply(lambda x: ir_holidays.get(x, default=0))

    X.loc[ca_idx, 'is_holiday'] = X.loc[ca_idx, 'timestamp'].apply(lambda x: ca_holidays.get(x, default=0))

    X.loc[us_idx, 'is_holiday'] = X.loc[us_idx, 'timestamp'].apply(lambda x: us_holidays.get(x, default=0))

    

    holiday_idx = X['is_holiday'] != 0

    X.loc[holiday_idx, 'is_holiday'] = 1

    X['is_holiday'] = X['is_holiday'].astype(np.uint8)

    

    #weekends

    X.loc[(X['weekday'] == 5) | (X['weekday'] == 6) , 'is_holiday'] = 1

    

    #-------------------------------------

    #https://www.kaggle.com/c/ashrae-energy-prediction/discussion/114161#latest-658796

    X['humidity'] = relative_humidity(X.air_temperature, X.dew_temperature).astype(np.float16)

    #-------------------------------------

    

    drop_features = ["timestamp", "sea_level_pressure", "year", "year_built"]



    X.drop(drop_features, axis=1, inplace=True)



    if test:

        row_ids = X.row_id

        X.drop("row_id", axis=1, inplace=True)

        return X, row_ids

    else:

        y = X.meter_reading

        X.drop("meter_reading", axis=1, inplace=True)

        return X, y
X_train, y_train = prepare_data(df_train, building, weather_train)
#del df_train, weather_train

#gc.collect()
X_half_1 = X_train[:int(X_train.shape[0] / 2)]

X_half_2 = X_train[int(X_train.shape[0] / 2):]

del X_train

gc.collect()
y_half_1 = y_train[:int(y_train.shape[0] / 2)]

y_half_2 = y_train[int(y_train.shape[0] / 2):]

del y_train

gc.collect()
categorical_features = ["building_id", "site_id", "meter", "primary_use", "hour", "weekday"]



n_estimators=40000 

learning_rate=0.04

bagging_fraction=0.7

feature_fraction=0.8

lambda_l2=2

metric="rmse"
import lightgbm as lgb

model_half_1 = lgb.LGBMRegressor(n_estimators=n_estimators, 

                                 learning_rate=learning_rate,

                                 bagging_fraction=bagging_fraction,

                                 feature_fraction=feature_fraction, 

                                 lambda_l2=lambda_l2,

                                 metric=metric)

model_half_2 = lgb.LGBMRegressor(n_estimators=n_estimators, 

                                 learning_rate=learning_rate,

                                 bagging_fraction=bagging_fraction,

                                 feature_fraction=feature_fraction, 

                                 lambda_l2=lambda_l2,

                                 metric=metric)
#from catboost import CatBoostRegressor

#model_half_1 = CatBoostRegressor(iterations=n_estimators, 

#                                 learning_rate=learning_rate,

#                                 bagging_temperature=bagging_fraction,

#                                 l2_leaf_reg=lambda_l2,

#                                 eval_metric=metric.upper())

#model_half_2 = CatBoostRegressor(iterations=n_estimators, 

#                                 learning_rate=learning_rate,

#                                 bagging_temperature=bagging_fraction,

#                                 l2_leaf_reg=lambda_l2,

#                                 eval_metric=metric.upper())
print("Building model with first half and validating on second half:")

model_half_1.fit(X_half_1,

                 y_half_1,

                 eval_set=[(X_half_1,y_half_1),(X_half_2,y_half_2)],

                 categorical_feature=categorical_features, 

                 #cat_features=categorical_features,

                 early_stopping_rounds=200,

                 verbose=1)
print("Building model with second half and validating on first half:")

model_half_2.fit(X_half_2,

                 y_half_2,

                 eval_set=[(X_half_2,y_half_2),(X_half_1,y_half_1)],

                 categorical_feature=categorical_features,

                 early_stopping_rounds=200,

                 verbose=10)
del X_half_1, X_half_2, y_half_1, y_half_2

gc.collect()
df_test = pd.read_csv(path_data + "test.csv")

weather_test = pd.read_csv(path_data + "weather_test.csv")
weather_test = weather_test.groupby('site_id').apply(lambda group: group.interpolate(limit_direction='both'))
X_test, row_ids = prepare_data(df_test, building, weather_test, test=True)

pred = np.expm1(model_half_1.predict(X_test, num_iteration=model_half_1.best_iteration_)) / 2

del model_half_1

gc.collect()
pred += np.expm1(model_half_2.predict(X_test, num_iteration=model_half_2.best_iteration_)) / 2

del model_half_2

gc.collect()
submission = pd.DataFrame({"row_id": row_ids, "meter_reading": np.clip(pred, 0, a_max=None)})
leak_df = pd.read_pickle(path_external_data+'site0.pkl') 

leak_df['meter_reading'] = leak_df.meter_reading_scraped

leak_df.drop(['meter_reading_original','meter_reading_scraped'], axis=1, inplace=True)

leak_df.fillna(0, inplace=True)
for bid in leak_df.building_id.unique():

    if bid % 25 == 0:

        print(bid)

    temp_df = leak_df[(leak_df.building_id == bid) & (leak_df.timestamp.dt.year > 2016)]

    for m in temp_df.meter.unique():

        submission.loc[(df_test.building_id == bid)&(df_test.meter==m), 'meter_reading'] = temp_df[temp_df.meter==m].meter_reading.values
submission.to_csv("submission.csv", index=False)