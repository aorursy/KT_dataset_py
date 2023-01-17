import pandas as pd

import numpy as np

import gc

import warnings

warnings.filterwarnings('ignore')



DATA_PATH = "../input/ashrae-energy-prediction/"



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



train_df = pd.read_csv(DATA_PATH + 'train.csv')

building_df = pd.read_csv(DATA_PATH + 'building_metadata.csv')

weather_df = pd.read_csv(DATA_PATH + 'weather_train.csv')



train_df = reduce_mem_usage(train_df,use_float16=True)

building_df = reduce_mem_usage(building_df,use_float16=True)

weather_df = reduce_mem_usage(weather_df,use_float16=True)



train_df = train_df.merge(building_df, left_on='building_id',right_on='building_id',how='left')

train_df = train_df.merge(weather_df,how='left',left_on=['site_id','timestamp'],right_on=['site_id','timestamp'])

del building_df,weather_df

gc.collect()

target = train_df["meter_reading"]

features = train_df.drop('meter_reading', axis = 1)

del train_df

gc.collect()
y_actual = np.array([2,4,6,8,10])

y_pred = np.array([2,5,6,7,10])



## Calculate RMSE - (R)sqrt->(M)mean>(Sqaure)power->(ERROR)loss

rmse = np.sqrt( np.mean(  np.power( (y_pred-y_actual) ,2) ) )

print("RMSE Score is {:.2f}".format(rmse))



## Calculate RMSLE - (R)sqrt->(M)mean>(Sqaure)power->(L)log->(ERROR)loss

rmsle = np.sqrt( np.mean(  np.power( (np.log1p(y_pred)-np.log1p(y_actual)) ,2) ) )

print("RMSLE Score is {:.2f}".format(rmsle))
target = np.log1p(target)
from lightgbm import LGBMRegressor

from sklearn.model_selection import cross_val_score



lightgbm = LGBMRegressor( 

    task = 'train',

    objective = "regression",

    boosting = "gbdt", 

    num_leaves = 40,

    learning_rate = 0.05,

    feature_fraction = 1,

    bagging_fraction = 1,

    lambda_l1 = 5,

    lambda_l2 = .1,

    max_depth = 5,

    min_child_weight = 1,

    min_split_gain = 0.001,

    num_boost_round=1,

    verbose= 100)



scores = cross_val_score(lightgbm, features, target, cv=3,scoring='neg_mean_squared_error')

# first convert to positive and then sqrt.

print("Average cross-validation RMSLE score:{:.2f}".format(np.sqrt(scores.mean()*-1)))

import lightgbm as lgb



train_data = lgb.Dataset(data=features, label=target, free_raw_data=False)

params = {}

params["task"] = 'train'

params["objective"] = 'regression'

params["boosting"] = 'gbdt'

params["num_leaves"] = 40

params['learning_rate'] = 0.05

params['feature_fraction'] = 1

params['bagging_fraction'] = 1

params['lambda_l1'] = 5

params['lambda_l2'] = .1

params['max_depth'] = 5

params['min_child_weight'] = 1

params['min_split_gain'] = 0.001

params['num_boost_round'] = 1

params['verbose'] = 100



cv_result = lgb.cv(params, train_data, nfold=3,metrics='rmse',stratified=False)

print("Average cross-validation RMSLE score:{:.2f}".format(np.min(cv_result['rmse-mean'])))