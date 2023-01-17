import numpy as np

import pandas as pd



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error

import xgboost as xgb



import gc

import os
BASE = '../input/ashrae-energy-prediction/'

building_df = pd.read_csv(BASE + "building_metadata.csv")

weather_train = pd.read_csv(BASE + "weather_train.csv")

train = pd.read_csv(BASE + "train.csv")
train = train.merge(building_df, left_on = "building_id", right_on = "building_id", how = "left")

train = train.merge(weather_train, left_on = ["site_id", "timestamp"], right_on = ["site_id", "timestamp"])

del weather_train

gc.collect()
train["primary_use"].value_counts()
le = LabelEncoder()

train["primary_use"] = le.fit_transform(train["primary_use"])
# https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65

def reduce_mem_usage(df):

    start_mem_usg = df.memory_usage().sum() / 1024**2 

    print("Memory usage of properties dataframe is :",start_mem_usg," MB")

    NAlist = [] # Keeps track of columns that have missing values filled in. 

    for col in df.columns:

        if df[col].dtype != object:  # Exclude strings            

            # Print current column type

#             print("******************************")

            print("Column: ",col)

            print("dtype before: ",df[col].dtype)            

            # make variables for Int, max and min

            IsInt = False

            mx = df[col].max()

            mn = df[col].min()

#             print("min for this col: ",mn)

#             print("max for this col: ",mx)

            # Integer does not support NA, therefore, NA needs to be filled

            if not np.isfinite(df[col]).all(): 

                NAlist.append(col)

                df[col].fillna(mn-1,inplace=True)  

                   

            # test if column can be converted to an integer

            asint = df[col].fillna(0).astype(np.int64)

            result = (df[col] - asint)

            result = result.sum()

            if result > -0.01 and result < 0.01:

                IsInt = True            

            # Make Integer/unsigned Integer datatypes

            if IsInt:

                if mn >= 0:

                    if mx < 255:

                        df[col] = df[col].astype(np.uint8)

                    elif mx < 65535:

                        df[col] = df[col].astype(np.uint16)

                    elif mx < 4294967295:

                        df[col] = df[col].astype(np.uint32)

                    else:

                        df[col] = df[col].astype(np.uint64)

                else:

                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:

                        df[col] = df[col].astype(np.int8)

                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:

                        df[col] = df[col].astype(np.int16)

                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:

                        df[col] = df[col].astype(np.int32)

                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:

                        df[col] = df[col].astype(np.int64)    

            # Make float datatypes 32 bit

            else:

                df[col] = df[col].astype(np.float32)

            

            # Print new column type

            print("dtype after: ",df[col].dtype)

#             print("******************************")

    # Print final result

    print("___MEMORY USAGE AFTER COMPLETION:___")

    mem_usg = df.memory_usage().sum() / 1024**2 

    print("Memory usage is: ",mem_usg," MB")

    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")

    return df, NAlist
train, _ = reduce_mem_usage(train)

gc.collect()
train["timestamp"] = pd.to_datetime(train["timestamp"])

train["weekend"] = train["timestamp"].dt.weekday.astype(np.uint8)

train["year"] = train["timestamp"].dt.year.astype(np.uint16)

train["month"] = train["timestamp"].dt.month.astype(np.uint8)

train["day"] = train["timestamp"].dt.day.astype(np.uint8)

train["hour"] = train["timestamp"].dt.hour.astype(np.uint8)

train.drop('timestamp', axis=1, inplace=True)

gc.collect()
target = np.log1p(train["meter_reading"])
categoricals = ["building_id", "primary_use", "hour", "day", "weekend", "month", "meter", "year"]

drop_cols = ["precip_depth_1_hr", "sea_level_pressure", "wind_direction", "wind_speed"]

numericals = ["square_feet", "year_built", "air_temperature", "cloud_coverage", "dew_temperature", "floor_count"]



feat_cols = categoricals + numericals
train.drop(drop_cols + ['site_id', 'meter_reading'], axis=1, inplace=True)

train = train[feat_cols]

gc.collect()
# kf = KFold(n_splits=5, shuffle=True, random_state=42)

# train_index, val_index = kf.split(train).__next__()

# X_train, y_train = train.iloc[train_index], target.iloc[train_index]

# X_val, y_val = train.iloc[val_index], target.iloc[val_index]

# model = xgb.XGBRegressor(n_estimators = 10, 

#                          max_depth=5, 

#                          subsample=.8, 

#                          learning_rate=0.1, 

#                          colsample_bytree=.8,

#                          objective='reg:squarederror'

#                         )



# model.fit(X_train, y_train,

#           eval_set=[(X_train, y_train), (X_val, y_val)],

#           eval_metric='rmse',

#           verbose=True)



kf = KFold(n_splits=5, shuffle=True, random_state=42)

train_index, val_index = kf.split(train).__next__()

X_train, y_train = train.iloc[train_index], target.iloc[train_index]

X_val, y_val = train.iloc[val_index], target.iloc[val_index]

xgb_train = xgb.DMatrix(X_train, y_train)

xgb_eval = xgb.DMatrix(X_val, y_val)

pars = {

    'colsample_bytree': 0.8,                 

    'learning_rate': 0.1,

    'max_depth': 5,

    'subsample': 0.8,

    'objective': 'reg:squarederror',

}

model = xgb.train(pars,

                  xgb_train,

                  num_boost_round=201,

                  evals=[(xgb_train, 'train'), (xgb_eval, 'val')],

                  verbose_eval=5,

                  early_stopping_rounds=20

                 )
del train, X_train, X_val, xgb_train, xgb_eval, y_train, y_val, target, kf

gc.collect()
test = pd.read_csv(BASE + "test.csv")

test = test.merge(building_df, left_on = "building_id", right_on = "building_id", how = "left")

del building_df

gc.collect()
weather_test = pd.read_csv(BASE + "weather_test.csv")

weather_test = weather_test.drop(drop_cols, axis = 1)

test = test.merge(weather_test, left_on = ["site_id", "timestamp"], right_on = ["site_id", "timestamp"], how = "left")

del weather_test

gc.collect()
test["primary_use"] = le.transform(test["primary_use"])
test, _ = reduce_mem_usage(test)

gc.collect()
test["timestamp"] = pd.to_datetime(test["timestamp"])

test["hour"] = test["timestamp"].dt.hour.astype(np.uint8)

test["day"] = test["timestamp"].dt.day.astype(np.uint8)

test["weekend"] = test["timestamp"].dt.weekday.astype(np.uint8)

test["month"] = test["timestamp"].dt.month.astype(np.uint8)

test["year"] = test["timestamp"].dt.year.astype(np.uint16)

test.drop(['timestamp', "site_id"], axis=1, inplace=True)

gc.collect()
test = test[feat_cols]
from tqdm import tqdm

i=0

res=[]

step_size = 50000

for j in tqdm(range(int(np.ceil(test.shape[0]/step_size)))):

    res.append(np.expm1(model.predict(xgb.DMatrix(test.iloc[i:i+step_size]))))

    i+=step_size
del test
res = np.concatenate(res)
sub = pd.read_csv(BASE+"sample_submission.csv")

sub["meter_reading"] = res

sub.to_csv("submission.csv", index = False)
# from IPython.display import FileLink

# FileLink('submission.csv')