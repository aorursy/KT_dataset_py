

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import gc

import pandas as pd

import numpy as np

import datetime

import matplotlib.pyplot as plt

from tqdm import tqdm  

from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.preprocessing import StandardScaler,LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import KFold,StratifiedKFold

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer



import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn
def fill_weather_dataset(weather_df, mode_type):

    

    weather_df.loc[:,'timestamp'] = weather_df['timestamp'].astype(str)

    # Find Missing Dates

    time_format = "%Y-%m-%d %H:%M:%S"

    start_date = datetime.datetime.strptime(weather_df['timestamp'].min(),time_format)

    end_date = datetime.datetime.strptime(weather_df['timestamp'].max(),time_format)

    total_hours = int(((end_date - start_date).total_seconds() + 3600) / 3600)

    hours_list = [(end_date - datetime.timedelta(hours=x)).strftime(time_format) for x in range(total_hours)]



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

    weather_df["hour"] = weather_df["datetime"].dt.hour

    weather_df["weekday"] = weather_df["datetime"].dt.weekday

#    

    #Use IterativeImputer to fill missing value 

#    df_weather_timestamp = weather_df.timestamp

#    weather_df = weather_df.drop(['timestamp','datetime'],axis=1)

#    imp = IterativeImputer(max_iter=20, random_state=0)

#    df_weather_train_np = imp.fit_transform(weather_df)

#    weather_df = pd.DataFrame(df_weather_train_np, columns=weather_df.columns)

#    weather_df.loc[:,'timestamp'] = df_weather_timestamp

    

    # Reset Index for Fast Update

    weather_df = weather_df.set_index(['site_id','day','month'])

    air_temperature_filler = pd.DataFrame(weather_df.groupby(['site_id','day','month'])['air_temperature'].mean(),columns=["air_temperature"])

    air_temperature_filler = air_temperature_filler.fillna(method='ffill')

    weather_df.update(air_temperature_filler,overwrite=False)



    # Step 1

    cloud_coverage_filler = weather_df.groupby(['site_id','day','month'])['cloud_coverage'].mean()

    # Step 2

    cloud_coverage_filler = pd.DataFrame(cloud_coverage_filler.fillna(method='ffill'),columns=["cloud_coverage"])



    weather_df.update(cloud_coverage_filler,overwrite=False)



    due_temperature_filler = pd.DataFrame(weather_df.groupby(['site_id','day','month'])['dew_temperature'].mean(),columns=["dew_temperature"])

    due_temperature_filler = pd.DataFrame(due_temperature_filler.fillna(method='ffill'),columns=["dew_temperature"])

    weather_df.update(due_temperature_filler,overwrite=False)



    # Step 1

    sea_level_filler = weather_df.groupby(['site_id','day','month'])['sea_level_pressure'].mean()

    # Step 2

    sea_level_filler = pd.DataFrame(sea_level_filler.fillna(method='ffill'),columns=['sea_level_pressure'])



    weather_df.update(sea_level_filler,overwrite=False)



    wind_direction_filler =  pd.DataFrame(weather_df.groupby(['site_id','day','month'])['wind_direction'].mean(),columns=['wind_direction'])

    wind_direction_filler =  pd.DataFrame(wind_direction_filler.fillna(method='ffill'),columns=["wind_direction"])

    weather_df.update(wind_direction_filler,overwrite=False)



    wind_speed_filler =  pd.DataFrame(weather_df.groupby(['site_id','day','month'])['wind_speed'].mean(),columns=['wind_speed'])

    wind_speed_filler =  pd.DataFrame(wind_speed_filler.fillna(method='ffill'),columns=["wind_speed"])

    weather_df.update(wind_speed_filler,overwrite=False)

     



    # Step 1

    precip_depth_filler = weather_df.groupby(['site_id','day','month'])['precip_depth_1_hr'].mean()

    # Step 2

    precip_depth_filler = pd.DataFrame(precip_depth_filler.fillna(method='ffill'),columns=['precip_depth_1_hr'])

    weather_df.update(precip_depth_filler,overwrite=False)

    if mode_type == 'Dnn':

        holidays = ["2016-01-01", "2016-01-18", "2016-02-15", "2016-05-30", "2016-07-04",

                    "2016-09-05", "2016-10-10", "2016-11-11", "2016-11-24", "2016-12-26",

                    "2017-01-01", "2017-01-16", "2017-02-20", "2017-05-29", "2017-07-04",

                    "2017-09-04", "2017-10-09", "2017-11-10", "2017-11-23", "2017-12-25",

                    "2018-01-01", "2018-01-15", "2018-02-19", "2018-05-28", "2018-07-04",

                    "2018-09-03", "2018-10-08", "2018-11-12", "2018-11-22", "2018-12-25",

                    "2019-01-01"] 

        weather_df["is_holiday"] = (weather_df.datetime.dt.date.astype("str").isin(holidays)).astype(int)

        

        beaufort = [(0, 0, 0.3), (1, 0.3, 1.6), (2, 1.6, 3.4), (3, 3.4, 5.5), (4, 5.5, 8), (5, 8, 10.8), (6, 10.8, 13.9), 

          (7, 13.9, 17.2), (8, 17.2, 20.8), (9, 20.8, 24.5), (10, 24.5, 28.5), (11, 28.5, 33), (12, 33, 200)]

        for item in beaufort:

            weather_df.loc[(weather_df['wind_speed']>=item[1]) & (weather_df['wind_speed']<item[2]), 'beaufort_scale'] = item[0]





 

    weather_df = weather_df.drop(['offset','datetime'],axis=1) 

    weather_df = weather_df.reset_index()     





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
def data_building_processing(df_data):

    '''===========Building data processing======================'''

    print('Processing building data...')



    lbl = LabelEncoder() 

    lbl.fit(list(df_data['primary_use'].values)) 

    df_data['primary_use'] = lbl.transform(list(df_data['primary_use'].values))

    imp = IterativeImputer(max_iter=30, random_state=0)

    df_build = imp.fit_transform(df_data)

    df_data = pd.DataFrame(df_build, columns=df_data.columns)

    df_data.loc[:,'floor_count'] = df_data['floor_count'].apply(int)

    df_data.loc[:,'year_built'] = df_data['year_built'].apply(int)  



#    df_data['year_built_1920'] = df_data['year_built'].apply(lambda x: 1 if x<1920 else 0 )

#    df_data['year_built_1920_1950'] = df_data['year_built'].apply(lambda x: 1 if 1920<=x & x<1950 else 0 )

#    df_data['year_built_1950_1970'] = df_data['year_built'].apply(lambda x: 1 if 1950<=x & x<1970 else 0 )

#    df_data['year_built_1970_2000'] = df_data['year_built'].apply(lambda x: 1 if 1970<=x & x<2000 else 0 )

#    df_data['year_built_2000'] = df_data['year_built'].apply(lambda x: 1 if x>=2000 else 0 )

    return df_data
def features_engineering(df, mode_type):

    



    classify_columns = ['building_id','meter','site_id','primary_use',

                        'hour','weekday'] 

    # Sort by timestamp

    df.sort_values("timestamp")

    df.reset_index(drop=True)

    

    # Add more features

    df['square_feet'] =  np.log1p(df['square_feet'])

    

    drop = ["timestamp","sea_level_pressure", "wind_direction", "wind_speed",]

    df = df.drop(drop, axis=1)

    gc.collect()

    

    # Encode Categorical Data

    for i in tqdm(classify_columns):

        le = LabelEncoder()

        df[i] = le.fit_transform(df[i])

        

    if mode_type == 'Dnn':

        numericals = ["square_feet", "year_built", "air_temperature", "cloud_coverage",

              "dew_temperature", "precip_depth_1_hr", "floor_count", 'beaufort_scale']

        print('Start working with numerical characteristics...')

        for i in tqdm(numericals):

            ss_X=StandardScaler() 

            df.loc[:,i] = ss_X.fit_transform(df[i].values.reshape(-1, 1))    

    

    return df
def data_pro(df_data, df_weather_train, df_weather_test, df_building, data_type='train',mode_type='lgb'):

    ## REducing memory

    df_data = reduce_mem_usage(df_data,use_float16=True)

    df_building = reduce_mem_usage(df_building,use_float16=True)

    '''===Align local timestamps===='''

    weather = pd.concat([df_weather_train,df_weather_test],ignore_index=True)

    weather_key = ['site_id', 'timestamp']

    temp_skeleton = weather[weather_key + ['air_temperature']].drop_duplicates(subset=weather_key).sort_values(by=weather_key).copy()

    # calculate ranks of hourly temperatures within date/site_id chunks

    temp_skeleton['temp_rank'] = temp_skeleton.groupby(['site_id', temp_skeleton.timestamp.dt.date])['air_temperature'].rank('average')

    # create a dataframe of site_ids (0-16) x mean hour rank of temperature within day (0-23)

    df_2d = temp_skeleton.groupby(['site_id', temp_skeleton.timestamp.dt.hour])['temp_rank'].mean().unstack(level=1)

    # Subtract the columnID of temperature peak by 14, getting the timestamp alignment gap.

    site_ids_offsets = pd.Series(df_2d.values.argmax(axis=1) - 14)

    site_ids_offsets.index.name = 'site_id'

    

    def timestamp_align(df):

        df['offset'] = df.site_id.map(site_ids_offsets)

        df['timestamp_aligned'] = (df.timestamp - pd.to_timedelta(df.offset, unit='H'))

        df['timestamp'] = df['timestamp_aligned']

        del df['timestamp_aligned']

        return df

 

    if data_type == 'test':

        print("Test data detected...")

        df_weather_test = timestamp_align(df_weather_test)

        df_weather_test = fill_weather_dataset(df_weather_test, mode_type)

        df_weather_test = reduce_mem_usage(df_weather_test,use_float16=True)

        #merge

        df_building = data_building_processing(df_building)

        df_data = pd.merge(df_data, df_building, on='building_id', how='left')

        df_data = df_data.merge(df_weather_test,how='left',left_on=['site_id','timestamp'],right_on=['site_id','timestamp'])

    

        print("Start feature processing...")    

        df_data = features_engineering(df_data, mode_type)

        gc.collect()

        

        return df_data

    

    elif data_type == 'train':

        print("Train data detected...")

        df_weather_train = timestamp_align(df_weather_train)

        df_weather_train = fill_weather_dataset(df_weather_train, mode_type)

        df_weather_train = reduce_mem_usage(df_weather_train,use_float16=True)

        #merge

        

        df_building = data_building_processing(df_building)

        df_data = df_data.merge(df_building, left_on='building_id',right_on='building_id',how='left')

        df_data = df_data.merge(df_weather_train,how='left',left_on=['site_id','timestamp'],right_on=['site_id','timestamp'])

        target_data = df_data['meter_reading']

        df_data = df_data.drop('meter_reading',axis=1)

        print("Start feature processing...")

        

        df_data = features_engineering(df_data, mode_type)

        gc.collect()



        return df_data, target_data
train_df = pd.read_csv('../input/ashrae-energy-prediction/train.csv')

train_df = train_df[train_df['building_id'] != 1099 ]

train_df = train_df.query('not (building_id <= 104 & meter == 0 & timestamp <= "2016-05-20")')

building_df = pd.read_csv('../input/ashrae-energy-prediction/building_metadata.csv')

weather_df = pd.read_csv("../input/ashrae-energy-prediction/weather_train.csv",parse_dates=["timestamp"],)

df_weather_test = pd.read_csv("../input/ashrae-energy-prediction/weather_test.csv",parse_dates=["timestamp"],)



train_data,target_data = data_pro(train_df, weather_df, df_weather_test, building_df, data_type='train')

target_data = np.log1p(target_data)



df_test = pd.read_csv('../input/ashrae-energy-prediction/test.csv')

row_ids = df_test["row_id"]

df_test.drop("row_id", axis=1, inplace=True)

building_df = pd.read_csv('../input/ashrae-energy-prediction/building_metadata.csv')

weather_df = pd.read_csv("../input/ashrae-energy-prediction/weather_train.csv",parse_dates=["timestamp"],) 

df_weather_test = pd.read_csv("../input/ashrae-energy-prediction/weather_test.csv",parse_dates=["timestamp"],)

test_data = data_pro(df_test, weather_df, df_weather_test, building_df, data_type='test')

test_data['row_id'] = row_ids

    
import lightgbm as lgb

params = {

    "objective": "regression",

    "boosting": "gbdt",

   'num_threads':16,

    'num_leaves': 1280,

#    'max_depth':10,

#    'min_data_in_leaf':20,

    "feature_fraction": 0.85,

#    'bagging_fraction':0.85,

    "reg_lambda": 2,

    "metric": "rmse",

#    'max_bin':255,

    'learning_rates':0.05

}



categorical_features = ["site_id", "building_id", "primary_use", "hour", "weekday",  "meter",'day','month','week',]

#                        'year_built_1920','year_built_1920_1950','year_built_1950_1970','year_built_1970_2000','year_built_2000'] 



kf = KFold(n_splits=3)

lgb_models = []

evals_results = []

for train_index,test_index in kf.split(train_data):

    train_features = train_data.loc[train_index]

    train_target = target_data.loc[train_index]

    

    test_features = train_data.loc[test_index]

    test_target = target_data.loc[test_index]

    

    d_training = lgb.Dataset(train_features, label=train_target,categorical_feature=categorical_features, free_raw_data=False)

    d_test = lgb.Dataset(test_features, label=test_target,categorical_feature=categorical_features, free_raw_data=False)

    

    evals_result = {}

    model = lgb.train(params, train_set=d_training,

                      num_boost_round=1000, 

                      valid_sets=[d_training,d_test], 

                      evals_result=evals_result,

                      verbose_eval=25, 

                      early_stopping_rounds=50)

    lgb_models.append(model)

    evals_results.append(evals_result)

    del train_features, train_target, test_features, test_target, d_training, d_test

    gc.collect()

    

    

print('plot...')            

for i in range(len(evals_results)):

    plt.figure()

    plt.subplots_adjust( wspace = 1, hspace = 0.01)

    ax = plt.subplot(1,2,1)

    lgb.plot_metric(evals_results[i], ax=ax)

    ax.set_title("model{}'s rmsle ".format(i))

    

    ax = plt.subplot(1,2,2)

    lgb.plot_importance(lgb_models[i], max_num_features=27, ax=ax)

    ax.set_title("model{}'s feature importance ".format(i))

    plt.show() 

    gc.collect()



del(train_df, weather_df, df_weather_test, building_df) 


#target_tests = []    

#for model in lgb_models:

#    if  target_tests == []:

#        target_tests = np.expm1(model.predict(test_data, num_iteration=model.best_iteration)) / len(lgb_models)

#    else:

#        target_tests += np.expm1(model.predict(test_data, num_iteration=model.best_iteration)) / len(lgb_models)

#    del model

#    gc.collect()



#results_df = pd.DataFrame({"row_id": row_ids, "meter_reading": np.clip(target_tests, 0, a_max=None)})

#del row_ids,target_tests,df_test,test_data

#gc.collect()

#results_df.to_csv("submission.csv", index=False)
