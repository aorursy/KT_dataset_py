import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime

import gc

import pickle

from sklearn.preprocessing import LabelEncoder

import sys

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error, mean_squared_error 

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold    

import matplotlib.pyplot as plt

import os

import lightgbm as lgb
## Function to reduce the DF size

# Original code from https://www.kaggle.com/gemartin/load-data-reduce-memory-usage by @gemartin

def reduce_mem_usage(df, verbose=True):

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

    return df
# Original imputing code for weather from https://www.kaggle.com/aitude/ashrae-missing-weather-data-handling by @aitude

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
def preprocessing(nsample = 0, type = 'train'):

    

    building_metadata = pd.read_csv('/kaggle/input/ashrae-energy-prediction/building_metadata.csv')



    if (type == 'train'):

        data = pd.read_csv('/kaggle/input/ashrae-energy-prediction/train.csv')

        weather = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_train.csv')           

    elif (type == 'test'):  

        data = pd.read_csv('/kaggle/input/ashrae-energy-prediction/test.csv')

        weather = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_test.csv')     



    weather = fill_weather_dataset(weather)



    data = reduce_mem_usage(data)

    building_metadata = reduce_mem_usage(building_metadata)

    weather = reduce_mem_usage(weather)



    # joining by building_id

    data = (building_metadata.set_index("building_id").join(data.set_index("building_id"))).reset_index()



    if (type == 'train'):

        # Correct units for site 0 to kwh    

        data.loc[(data['site_id'] == 0) & (data['meter'] == 0), 'meter_reading'] = data[(data['site_id'] == 0) & (data['meter'] == 0)]['meter_reading'] * 0.2931    

    

    # joining by site_id and timestamp using multi indexes

    data = data.set_index(['site_id','timestamp']).join(weather.set_index(['site_id','timestamp'])).reset_index()

    del building_metadata, weather

    gc.collect()

    

    if nsample > 0 :

        data = data.sample(n=nsample)

        gc.collect()

    

    # Convert timestamp string to datetime

    data.loc[:, 'timestamp'] = pd.to_datetime(data.timestamp)

    data['month'] = pd.DatetimeIndex(data.timestamp).month

    data['weekday'] = pd.DatetimeIndex(data.timestamp).dayofweek

    data['hour'] = pd.DatetimeIndex(data.timestamp).hour

    data['day'] = pd.DatetimeIndex(data.timestamp).day



    if (type == 'train'):    

        # Remove outliers

        Meter1_Outliers = data.loc[(data.meter == 1) & (data.meter_reading > 20000)].building_id.unique()

        data = data[~data['building_id'].isin(Meter1_Outliers)] 

        Meter2_Outliers = data.loc[(data.meter == 2) & (data.meter_reading > 20000)].building_id.unique()

        data = data[~data['building_id'].isin(Meter2_Outliers)] 

        Meter3_Outliers = data.loc[(data.meter == 3) & (data.meter_reading > 5000)].building_id.unique()

        data = data[~data['building_id'].isin(Meter3_Outliers)] 



        # Remove all rows where the meter reading is 0

        data = data.drop(data.loc[data.meter_reading == 0].index, axis = 0)

        # Split target and take the log (because the evaluation metric is RMSLE 

        # and we'll use the RMSE metric for training)

        y = np.log1p(data.meter_reading)

        data = data.drop('meter_reading', axis = 1)           

    elif (type == 'test'):    

        y = data.row_id

        data = data.drop('row_id', axis = 1)

    

    # Dropping useless

    useless = ['timestamp', "sea_level_pressure", "wind_direction", "wind_speed","year_built","floor_count"]

    data = data.drop(useless, axis = 1)

    gc.collect()

    

    pkl_file = open('/kaggle/input/ashrae-preprocessing-train/LabelEncoder.pkl', 'rb')

    le = pickle.load(pkl_file)

    data["primary_use"] = le.transform(data["primary_use"])



    data = reduce_mem_usage(data)   

    print(data.memory_usage().sum() / 1024**2, 'Mb')    

        

    return data, y
preprocessingtask = 'load'               # 'load' saved data from preprocessing notebooks

                                        #  or 'compute' fresh data (much longer)

if (preprocessingtask == 'load') :

    pkl_file = open('/kaggle/input/ashrae-preprocessing-train/data_train.pkl', 'rb')

    data_train = pickle.load(pkl_file)

    pkl_file.close()

    pkl_file = open('/kaggle/input/ashrae-preprocessing-train/y.pkl', 'rb')

    y = np.log1p(pickle.load(pkl_file))

    pkl_file.close()

    pkl_file = open('/kaggle/input/ashrae-preprocessing-test/data_test.pkl', 'rb')

    data_test = pickle.load(pkl_file)

    pkl_file.close()

    pkl_file = open('/kaggle/input/ashrae-preprocessing-test/row_id.pkl', 'rb')

    row_id = pickle.load(pkl_file)

    pkl_file.close()

elif (preprocessingtask == 'compute') :

    data_train, y = preprocessing(nsample = 0, type = 'train')    
gc.collect()
numerical_cols = ['square_feet', 'air_temperature','dew_temperature', 'precip_depth_1_hr']

categorical_cols = ['site_id', 'building_id','primary_use', 'meter', 'cloud_coverage','month', 'weekday', 'hour', 'day']



features_cols = numerical_cols + categorical_cols
data_train, X_valid, y, y_valid = train_test_split(data_train, y, train_size=0.8, test_size=0.2)
# Sample grid search for hyperparameter optimisation 

# (inactive by default since it has already been done)



GridSearch = False



if GridSearch :

    search_parameters = {'learning_rate' : [0.25, 0.3, 0.35],

                         'num_leaves' : [900, 1000, 1250],

                         'feature_fraction' : [0.85],   

                         'reg_lambda': [0.7, 0.8, 1],}

    clf = lgb.LGBMRegressor()

    gs = GridSearchCV(

        estimator=clf, 

        param_grid=search_parameters, 

        cv=3,

        refit=True,

        verbose=True)



    search = gs.fit(X_train, y_train)

    print(search.best_params_)

    Lgb_predictions = Lgb_predictions = gs.predict(X_valid)
if ~GridSearch :  # Main model definition and training      

    params = {  'boosting_type': 'gbdt',

                'objective': 'regression',

                'metric': 'rmse',

                'categorical_features' : categorical_cols,

                'feature_fraction': 0.85, 

                'learning_rate': 0.35, 

                'num_leaves': 1250, 

                'reg_lambda': 0.8,

            }



    # Using 3-fold training 

    kf = KFold(n_splits=3)

    models = []

    

    y = y.reset_index()

    y = y.drop('index', axis = 1)

    data_train = data_train.reset_index()

    data_train = data_train.drop('index', axis = 1)



    for train_index,test_index in kf.split(data_train):       

        train_features = data_train.loc[train_index]

        train_target = y.loc[train_index]

        test_features = data_train.loc[test_index]    

        test_target = y.loc[test_index]



        d_training = lgb.Dataset(train_features, label=train_target,categorical_feature=categorical_cols, free_raw_data=False)

        d_test = lgb.Dataset(test_features, label=test_target,categorical_feature=categorical_cols, free_raw_data=False)

    

        model = lgb.train(params, train_set=d_training, num_boost_round = 1000, valid_sets=[d_training,d_test], verbose_eval=25, early_stopping_rounds=50)

        models.append(model)

        del train_features, train_target, test_features, test_target, d_training, d_test

        gc.collect()        
# Final validation on the hold-out set

Lgb_predictions = []  

for model in models:

    if  Lgb_predictions == []:

        # taking the exp to revert the earlier log  

        Lgb_predictions = np.expm1(model.predict(X_valid, num_iteration=model.best_iteration)) / len(models)

    else:

        Lgb_predictions += np.expm1(model.predict(X_valid, num_iteration=model.best_iteration)) / len(models)  

        

# Reverting unit for scoring in site 0 : * 3.4118

X_valid['row_id'] = np.arange(len(X_valid))

site0_idx = X_valid.loc[(X_valid.site_id == 0) & (X_valid.meter == 0)].index

to_correct = X_valid.loc[site0_idx].row_id

Lgb_predictions[to_correct] = Lgb_predictions[to_correct] * 3.4118

X_valid = X_valid.drop('row_id', axis = 1)

    

print(mean_absolute_error(y_valid, Lgb_predictions))

print(mean_squared_error(y_valid, Lgb_predictions))

gc.collect()
# Feature importance by model

import matplotlib.pyplot as plt



for model in models:

    lgb.plot_importance(model)

    plt.show()
output = open('models.pkl', 'wb')  

pickle.dump(models, output)

output.close()