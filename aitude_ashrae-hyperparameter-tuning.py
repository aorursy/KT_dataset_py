import pandas as pd

import numpy as np

import lightgbm as lgb

from sklearn.preprocessing import LabelEncoder

import datetime

import gc

from bayes_opt import BayesianOptimization

import warnings

warnings.filterwarnings('ignore')



DATA_PATH = "../input/ashrae-energy-prediction/"
train_df = pd.read_csv(DATA_PATH + 'train.csv',nrows=1000000)

train_df = train_df [ train_df['building_id'] != 1099 ]

train_df = train_df.query('not (building_id <= 104 & meter == 0 & timestamp <= "2016-05-20")')

building_df = pd.read_csv(DATA_PATH + 'building_metadata.csv')

weather_df = pd.read_csv(DATA_PATH + 'weather_train.csv')
# Original code from https://www.kaggle.com/aitude/ashrae-missing-weather-data-handling by @aitude



def fill_weather_dataset(weather_df):

    

    # Find Missing Dates

    time_format = "%Y-%m-%d %H:%M:%S"

    start_date = datetime.datetime.strptime(weather_df['timestamp'].min(),time_format)

    end_date = datetime.datetime.strptime(weather_df['timestamp'].max(),time_format)

    total_hours = int(((end_date - start_date).total_seconds() + 3600) / 3600)

    hours_list = [(end_date - datetime.timedelta(hours=x)).strftime(time_format) for x in range(total_hours)]



    missing_hours = []

    for site_id in range(16):

        site_hours = np.array(weather_df[weather_df['site_id'] == site_id]['timestamp'])

        new_rows = pd.DataFrame(np.setdiff1d(hours_list,site_hours),columns=['timestamp'])

        new_rows['site_id'] = site_id

        weather_df = pd.concat([weather_df,new_rows])



        weather_df = weather_df.reset_index(drop=True)           



    # Add new Features

    weather_df["datetime"] = pd.to_datetime(weather_df["timestamp"])

    weather_df["day"] = weather_df["datetime"].dt.day

    weather_df["week"] = weather_df["datetime"].dt.week

    weather_df["month"] = weather_df["datetime"].dt.month

    

    # Reset Index for Fast Update

    weather_df = weather_df.set_index(['site_id','day','month'])



    air_temperature_filler = pd.DataFrame(weather_df.groupby(['site_id','day','month'])['air_temperature'].mean(),columns=["air_temperature"])

    weather_df.update(air_temperature_filler,overwrite=False)



    # Step 1

    cloud_coverage_filler = weather_df.groupby(['site_id','day','month'])['cloud_coverage'].mean()

    # Step 2

    cloud_coverage_filler = pd.DataFrame(cloud_coverage_filler.fillna(method='ffill'),columns=["cloud_coverage"])



    weather_df.update(cloud_coverage_filler,overwrite=False)



    due_temperature_filler = pd.DataFrame(weather_df.groupby(['site_id','day','month'])['dew_temperature'].mean(),columns=["dew_temperature"])

    weather_df.update(due_temperature_filler,overwrite=False)



    # Step 1

    sea_level_filler = weather_df.groupby(['site_id','day','month'])['sea_level_pressure'].mean()

    # Step 2

    sea_level_filler = pd.DataFrame(sea_level_filler.fillna(method='ffill'),columns=['sea_level_pressure'])



    weather_df.update(sea_level_filler,overwrite=False)



    wind_direction_filler =  pd.DataFrame(weather_df.groupby(['site_id','day','month'])['wind_direction'].mean(),columns=['wind_direction'])

    weather_df.update(wind_direction_filler,overwrite=False)



    wind_speed_filler =  pd.DataFrame(weather_df.groupby(['site_id','day','month'])['wind_speed'].mean(),columns=['wind_speed'])

    weather_df.update(wind_speed_filler,overwrite=False)



    # Step 1

    precip_depth_filler = weather_df.groupby(['site_id','day','month'])['precip_depth_1_hr'].mean()

    # Step 2

    precip_depth_filler = pd.DataFrame(precip_depth_filler.fillna(method='ffill'),columns=['precip_depth_1_hr'])



    weather_df.update(precip_depth_filler,overwrite=False)



    weather_df = weather_df.reset_index()

    weather_df = weather_df.drop(['datetime','day','week','month'],axis=1)

        

    return weather_df



# Original code from https://www.kaggle.com/gemartin/load-data-reduce-memory-usage by @gemartin



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





def features_engineering(df):

    

    # Sort by timestamp

    df.sort_values("timestamp")

    df.reset_index(drop=True)

    

    # Add more features

    df["timestamp"] = pd.to_datetime(df["timestamp"],format="%Y-%m-%d %H:%M:%S")

    df["hour"] = df["timestamp"].dt.hour

    df["weekend"] = df["timestamp"].dt.weekday

    holidays = ["2016-01-01", "2016-01-18", "2016-02-15", "2016-05-30", "2016-07-04",

                    "2016-09-05", "2016-10-10", "2016-11-11", "2016-11-24", "2016-12-26",

                    "2017-01-02", "2017-01-16", "2017-02-20", "2017-05-29", "2017-07-04",

                    "2017-09-04", "2017-10-09", "2017-11-10", "2017-11-23", "2017-12-25",

                    "2018-01-01", "2018-01-15", "2018-02-19", "2018-05-28", "2018-07-04",

                    "2018-09-03", "2018-10-08", "2018-11-12", "2018-11-22", "2018-12-25",

                    "2019-01-01"]

    df["is_holiday"] = (df.timestamp.isin(holidays)).astype(int)

    df['square_feet'] =  np.log1p(df['square_feet'])

    

    # Remove Unused Columns

    drop = ["timestamp","sea_level_pressure", "wind_direction", "wind_speed","year_built","floor_count"]

    df = df.drop(drop, axis=1)

    gc.collect()

    

    # Encode Categorical Data

    le = LabelEncoder()

    df["primary_use"] = le.fit_transform(df["primary_use"])

    

    return df
weather_df = fill_weather_dataset(weather_df)
train_df = reduce_mem_usage(train_df,use_float16=True)

building_df = reduce_mem_usage(building_df,use_float16=True)

weather_df = reduce_mem_usage(weather_df,use_float16=True)
train_df = train_df.merge(building_df, left_on='building_id',right_on='building_id',how='left')

train_df = train_df.merge(weather_df,how='left',left_on=['site_id','timestamp'],right_on=['site_id','timestamp'])

del weather_df

gc.collect()
train_df = features_engineering(train_df)
target = np.log1p(train_df["meter_reading"])

features = train_df.drop('meter_reading', axis = 1)

del train_df

gc.collect()
def lgb_best_params(X, y,opt_params,init_points=2, optimization_round=20, n_folds=3, random_seed=0, cv_estimators=1000):

    

    # prepare dataset

    categorical_features = ["building_id", "site_id", "meter", "primary_use", "is_holiday", "weekend"]

    train_data = lgb.Dataset(data=X, label=y, categorical_feature = categorical_features, free_raw_data=False)

    

    def lgb_run(num_leaves, feature_fraction, bagging_fraction, max_depth, lambda_l1, lambda_l2, min_split_gain, min_child_weight,learning_rate):

        params = {"boosting": "gbdt",'application':'regression','num_iterations':cv_estimators, 'early_stopping_round':int(cv_estimators/5), 'metric':'rmse'}

        params["num_leaves"] = int(round(num_leaves))

        params['feature_fraction'] = max(min(feature_fraction, 1), 0)

        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)

        params['max_depth'] = int(round(max_depth))

        params['lambda_l1'] = max(lambda_l1, 0)

        params['lambda_l2'] = max(lambda_l2, 0)

        params['min_split_gain'] = min_split_gain

        params['min_child_weight'] = min_child_weight

        params['learning_rate'] = learning_rate

        cv_result = lgb.cv(params, train_data, nfold=n_folds, seed=random_seed, stratified=False, verbose_eval =cv_estimators, metrics=['rmse'])

        return -min(cv_result['rmse-mean'])

    

    params_finder = BayesianOptimization(lgb_run, opt_params, random_state=2021)

    # optimize

    params_finder.maximize(init_points=init_points, n_iter=optimization_round)



    # return best parameters

    return params_finder.max
#  number of fold

fold = 3



# Hyperparameter range

params_range = {

                'num_leaves': (1000, 1280),

                'feature_fraction': (0.7, 0.9),

                'bagging_fraction': (0.8, 1),

                'max_depth': (10, 11),

                'lambda_l1': (2, 5),

                'lambda_l2': (2, 5),

                'min_split_gain': (0.001, 0.1),

                'min_child_weight': (5, 50),

                'learning_rate' : (.05,.07)

               }



# You can experiments with different estimators in a single execution. I'm using small numbers for demonstration purpose so you must change to high numbers.

cv_estimators = [50,100,200]



#n_iter: How many steps of bayesian optimization you want to perform. The more steps the more likely to find a good maximum you are.

optimization_round = 10 # Simply, 10 models with different parameter will be tested. And the best one will be returned.



#init_points: How many steps of random exploration you want to perform. Random exploration can help by diversifying the exploration space.

init_points = 2



random_seed = 2010
best_params= []

for cv_estimator in cv_estimators:

    opt_params = lgb_best_params(features, target,params_range, init_points=init_points, optimization_round=optimization_round, n_folds=fold, random_seed=random_seed, cv_estimators=cv_estimator)

    opt_params['params']['iteration'] = cv_estimator

    opt_params['params']['fold'] = fold

    opt_params['params']['rmse'] = opt_params['target']

    best_params.append(opt_params['params'])
df_params = pd.DataFrame(best_params).reset_index()

df_params = df_params[['iteration','fold','num_leaves','learning_rate','bagging_fraction',

 'feature_fraction',

 'lambda_l1',

 'lambda_l2',

 'max_depth',

 'min_child_weight',

 'min_split_gain',

 'rmse']]

df_params.to_csv('best_params.csv')
df_params.head()