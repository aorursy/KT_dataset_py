import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn

import matplotlib.pyplot as plt

import gc

from sklearn.preprocessing import StandardScaler

# taken from https://www.kaggle.com/aitude/ashrae-missing-weather-data-handling

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
path = "/kaggle/input/ashrae-energy-prediction/"
data_dtype = {'meter':"uint8",'building_id':'uint16'}



train = pd.read_csv(path + "train.csv", delimiter=",", dtype=data_dtype)

test = pd.read_csv(path + "test.csv", delimiter=",", dtype=data_dtype)
metadata_dtype = {'site_id':"uint8",'building_id':'uint16','square_feet':'float32','year_built':'float32','floor_count':"float16"}

weather_dtype = {"site_id":"uint8"}



weather_train = pd.read_csv(path + "weather_train.csv", delimiter=",", dtype=weather_dtype)

weather_test = pd.read_csv(path + "weather_test.csv", delimiter=",", dtype=weather_dtype)



metadata = pd.read_csv(path + "building_metadata.csv", delimiter=",", dtype=metadata_dtype)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

metadata.primary_use = le.fit_transform(metadata.primary_use)

gc.collect()
import datetime

def fill_weather_missing_rows(weather_data):

    time_format = "%Y-%m-%d %H:%M:%S"

    start_date = datetime.datetime.strptime(weather_data['timestamp'].min(),time_format)

    end_date = datetime.datetime.strptime(weather_data['timestamp'].max(),time_format)

    total_hours = int(((end_date - start_date).total_seconds() + 3600) / 3600)

    hours_list = [(end_date - datetime.timedelta(hours=x)).strftime(time_format) for x in range(total_hours)]

    missing_hours = []

    for site_id in range(16):

        

        site_hours = np.array(weather_data[weather_data['site_id'] == site_id]['timestamp'])

        new_rows = pd.DataFrame(np.setdiff1d(hours_list, site_hours),columns=['timestamp'])

        new_rows['site_id'] = site_id

        weather_data = pd.concat([weather_data, new_rows])

    return weather_data.reset_index(drop=True) 



def extend_timestamp_weather(df):

    df["datetime"] = pd.to_datetime(df["timestamp"])

    df["day"] = df["datetime"].dt.day

    df["week"] = df["datetime"].dt.week

    df["month"] = df["datetime"].dt.month

    return df.set_index(['site_id','day','month'])



def fill_weather_missing_values(df):    

    missing_hours = ["air_temperature","dew_temperature","wind_direction", "wind_speed"]

    missing_days = ["cloud_coverage", "sea_level_pressure", "precip_depth_1_hr"]

    

    for col in missing_hours:

        filler = pd.DataFrame(df.groupby(['site_id','day','month'])[col].mean(),columns=[col])

        df.update(filler,overwrite=False) 

        

    for col in missing_days:

        filler = df.groupby(['site_id','day','month'])[col].mean()

        filler = pd.DataFrame(filler.fillna(method='ffill'),columns=[col])

        df.update(filler,overwrite=False)

        #'datetime','day','week','month'

    df = df.reset_index().drop(['datetime','day','week','month'],axis=1)   



    return df



def fill_weather(df):

    return fill_weather_missing_values(extend_timestamp_weather(fill_weather_missing_rows(df)))
def merge_sets(data, metadata, weather):

    data = pd.merge(data, metadata, on='building_id',how='left')

    gc.collect()

    data = pd.merge(data, weather,how='left',on=['site_id','timestamp'])

    gc.collect()

    return data
def engineer_metadata(data=metadata):

    data.drop("floor_count", axis=1, inplace=True)

    data['year_built'].fillna(-999, inplace=True)

    data['year_built'] = data['year_built'] - 1900

    return data
weather_train = fill_weather(weather_train)

metadata = engineer_metadata()

train = merge_sets(train, metadata, weather_train)

del weather_train

gc.collect()



weather_test = fill_weather(weather_test)

test = merge_sets(test, metadata, weather_test)

del weather_test, metadata

reduce_mem_usage(test)

gc.collect()
def feature_engineering(data=train, cat_cols=None, drop_meter_readings=False, drop_outliers=False, drop_zero_el=False, cols_to_drop=None, log_cols=[], std_cols=None, shuffle_train=False):   

    # https://www.kaggle.com/c/ashrae-energy-prediction/discussion/117083

    if drop_zero_el:

        idx_to_drop = list(data[(data['meter'] == "Electricity") & (data['meter_reading'] == 0)].index)

        data.drop(idx_to_drop,axis='rows',inplace=True)



    if drop_meter_readings:

        idx_to_drop = list((train[(data['site_id'] == 0) & (data['timestamp'] < "2016-05-21 00:00:00")]).index)

        data.drop(idx_to_drop,axis='rows',inplace=True)

      

    # got the idea to include holidays from here https://www.kaggle.com/rohanrao/ashrae-half-and-half

    holidays = ["2016-01-01", "2016-01-18", "2016-02-15", "2016-05-30", "2016-07-04",

            "2016-09-05", "2016-10-10", "2016-11-11", "2016-11-24", "2016-12-26",

            "2017-01-01", "2017-01-16", "2017-02-20", "2017-05-29", "2017-07-04",

            "2017-09-04", "2017-10-09", "2017-11-10", "2017-11-23", "2017-12-25",

            "2018-01-01", "2018-01-15", "2018-02-19", "2018-05-28", "2018-07-04",

            "2018-09-03", "2018-10-08", "2018-11-12", "2018-11-22", "2018-12-25",

            "2019-01-01"]

    

    data["timestamp"] = pd.to_datetime(data["timestamp"])

    data["is_holiday"] = (data.timestamp.dt.date.astype("str").isin(holidays)).astype(int)

    data['month'] = data['timestamp'].dt.month.astype("uint8")

    data['day'] = data['timestamp'].dt.day.astype("uint8")

    data['weekday'] = data['timestamp'].dt.dayofweek.astype("uint8")

    data['hour'] = data['timestamp'].dt.hour.astype("uint8")

    gc.collect()

    

    # IIRC I got the idea for that in here https://www.kaggle.com/starl1ght/ashrae-stacked-regression-lasso-ridge-lgbm

    if drop_outliers:

        data = data[data['building_id'] != 1099 ]

        data = data.query('not (building_id <= 104 & meter == 0 & timestamp <= "2016-05-20")')

    gc.collect()

    

    if cols_to_drop is not None:

        data.drop(cols_to_drop, inplace=True, axis=1)

    

    for col in log_cols:

        data[col] = np.log1p(data[col])

        

    # ended up not using this

    if std_cols is not None:

        scaler = StandardScaler()

        scaler.fit(train[std_cols])

        data[std_cols] = scaler.transform(data[std_cols])

    

    # ended up not using this

    if cat_cols is not None:

        data = pd.get_dummies(data, columns=categorical_cols, sparse=True)

    



    gc.collect()

    return data
def splitXy(data):

    return data.drop("meter_reading",axis=1), data.meter_reading
categorical_cols = ['building_id','month','meter','hour','primary_use','weekday','day']
X_tr, y_tr = splitXy(feature_engineering(

    drop_meter_readings=True,

    drop_zero_el=True, 

    drop_outliers=True,

    cols_to_drop=["timestamp","sea_level_pressure", "wind_direction", "wind_speed","year_built"], 

    log_cols=['square_feet', "meter_reading"],

    std_cols=None

))

del train

gc.collect()
row_ids = test.row_id

X_te = feature_engineering(

    data=test,

    drop_meter_readings=False,

    drop_zero_el=False, 

    drop_outliers=False,

    cols_to_drop=["timestamp","sea_level_pressure", "wind_direction", "wind_speed","year_built"], 

    log_cols=['square_feet'],

    std_cols=None

)

del test

gc.collect()
import lightgbm as lgb

from sklearn.model_selection import KFold



params = {"objective": "regression",

                  "num_leaves": 40,

                  "learning_rate": 0.05,

                  "boosting": "gbdt",

                  "bagging_freq": 5,

                  "feature_fraction": 0.85,

                  "bagging_fraction": 0.51,

                  "metric": "rmse"

                  }



for i in range(16):

    print("----------------")

    print("Model", i)

    idx = (X_tr.site_id == i)

    X = X_tr.loc[idx].drop("site_id", axis=1)

    y = y_tr.loc[idx]

    

    del idx

    gc.collect() 

    

    kf = KFold(n_splits=3)

    models = []

    for train_index,test_index in kf.split(X):

        train_features = X.iloc[train_index]

        train_target = y.iloc[train_index]



        test_features = X.iloc[test_index]

        test_target = y.iloc[test_index]



        d_training = lgb.Dataset(train_features, label=train_target,categorical_feature=categorical_cols, free_raw_data=False)

        d_test = lgb.Dataset(test_features, label=test_target,categorical_feature=categorical_cols, free_raw_data=False)



        model = lgb.train(params, train_set=d_training, num_boost_round=1000, valid_sets=[d_training,d_test], verbose_eval=25, early_stopping_rounds=50)

        models.append(model)

        del train_features, train_target, test_features, test_target, d_training, d_test

        gc.collect()

        

    del X, y

    gc.collect() 

    

    print("Predicting...")

    idx = (X_te.site_id == i)

    test_features = X_te.loc[idx].drop(["site_id", "row_id"], axis=1)

    

    results = []

    for model in models:

        if  results == []:

            results = np.expm1(model.predict(test_features, num_iteration=model.best_iteration)) / len(models)

        else:

            results += np.expm1(model.predict(test_features, num_iteration=model.best_iteration)) / len(models)

        del model

        gc.collect()

    del models  

    gc.collect()

    

    results = pd.DataFrame({"row_id": X_te.row_id[idx], "meter_reading": np.clip(results, 0, a_max=None)})

    results.to_csv('results.csv', mode='a', header=False)

    del results, test_features, idx

    gc.collect() 

    

del X_tr, y_tr, X_te

gc.collect()
df = pd.read_csv("results.csv", names=['row_id', 'meter_reading'], header=None)
df.to_csv("submission.csv", index=False)