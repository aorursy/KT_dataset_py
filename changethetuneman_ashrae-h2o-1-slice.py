import gc

import os

import random



import lightgbm as lgb

import numpy as np

import pandas as pd

import seaborn as sns

from pathlib import Path



from matplotlib import pyplot as plt

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import LabelEncoder

from IPython.display import FileLink



path_data = "/kaggle/input/ashrae-energy-prediction/"

path_train = path_data + "train.csv"

path_test = path_data + "test.csv"

path_building = path_data + "building_metadata.csv"

path_weather_train = path_data + "weather_train.csv"

path_weather_test = path_data + "weather_test.csv"



plt.style.use("seaborn")

sns.set(font_scale=1)



myfavouritenumber = 0

seed = myfavouritenumber

random.seed(seed)

pd.set_option('display.max_columns', 50)

pd.set_option('display.max_rows', 150)
%%time

root = Path('../input/ashrae-feather-format-for-fast-loading')



train_df = pd.read_feather(root/'train.feather')

weather_train_df = pd.read_feather(root/'weather_train.feather')

building_meta_df = pd.read_feather(root/'building_metadata.feather')

from pandas.api.types import is_datetime64_any_dtype as is_datetime

from pandas.api.types import is_categorical_dtype



def reduce_mem_usage(df, use_float16=False):

    """

    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.

    """

    start_mem = df.memory_usage().sum() / 1024 ** 2

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



    end_mem = df.memory_usage().sum() / 1024 ** 2

    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))

    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))



    return df
import numpy as np

import pandas as pd

import datetime, math





def preprocess(train, test=False):



    def primary_use(x):

        p_use = {"Religious worship":3.91,"Retail":26.61,"Other":35.10,"Warehouse/storage":70.93,"Technology/science":227.89,

                 "Food sales and service":304.76,"Lodging/residential":307.98,"Entertainment/public assembly":320.39,"Parking":321.10,

                 "Public services":373.27,"Utility":538.77,"Manufacturing/industrial":549.41,"Office":752.07,"Healthcare":820.03,

                 "Education":2456.70,"Services":10026.19}

        for key in p_use.keys():

            if x == key: return p_use.get(key)

    train['primary_use']=train['primary_use'].apply(lambda x: primary_use(x))

    # floor_count

    train['floor_count_ifnan'] = train.floor_count.isnull().astype('int')



    # dew_temperature

    train['dew_temperature'] = train['dew_temperature'].fillna(23)

    train['dew_temperature_k'] = train['dew_temperature'].apply(lambda x: x + 273.15)



    # air_temperature

    train["air_temperature"] = train["air_temperature"].fillna(35)

    train['air_temperature_k'] = train['air_temperature'].apply(lambda x: x + 273.15)



    # precip_depth_1_hr

    train['precip_depth_1_hr_ifnan'] = train.precip_depth_1_hr.isnull().astype("int")

    train['precip_depth_1_hr'] = train['precip_depth_1_hr'].fillna(300)



    # sea_level_pressure

    train['sea_level_pressure'] = train['sea_level_pressure'].fillna(980)

    train['sea_level_pressure_atm'] = train['sea_level_pressure'].apply(lambda x: x / 1013.25)



    # wind_direction

    train['wind_direction'] = train['wind_direction'].fillna(0)



    # wind_speed

    train['wind_speed'] = train['wind_speed'].fillna(15)



    # timestamp

    train.timestamp = pd.to_datetime(train.timestamp, format="%Y-%m-%d %H:%M:%S")

    train.square_feet = np.log1p(train.square_feet)



    if not test:

        train.sort_values("timestamp", inplace=True)

        train.reset_index(drop=True, inplace=True)



    holidays = ["2016-01-01", "2016-01-18", "2016-02-15", "2016-05-30", "2016-07-04",

                "2016-09-05", "2016-10-10", "2016-11-11", "2016-11-24", "2016-12-26",

                "2017-01-01", "2017-01-16", "2017-02-20", "2017-05-29", "2017-07-04",

                "2017-09-04", "2017-10-09", "2017-11-10", "2017-11-23", "2017-12-25",

                "2018-01-01", "2018-01-15", "2018-02-19", "2018-05-28", "2018-07-04",

                "2018-09-03", "2018-10-08", "2018-11-12", "2018-11-22", "2018-12-25",

                "2019-01-01"]



    # year_build

    train['year_built_ifnan'] = train.year_built.isnull().astype('int')

    train['year_built'] = train['year_built'].fillna(2015)



    train["hour"] = train.timestamp.dt.hour

    train['year'] = train['timestamp'].dt.year

    train['month'] = train['timestamp'].dt.month

    train['day'] = train['timestamp'].dt.day

    train["weekday"] = train.timestamp.dt.weekday

    train['age'] = (train['year'] - train['year_built'])

    train["is_holiday"] = (train.timestamp.dt.date.astype("str").isin(holidays)).astype(int)



    # relative_humidity

    train['relative_humidity'] = 100 - 5 * (train['air_temperature_k'] - train['dew_temperature_k'])



    # air_density

    train['air_density'] = (train['sea_level_pressure_atm'] * 14.67) / (train['air_temperature_k'] * 0.0821)



    # saturated_vapour_density

    vapour_density = 7.2785

    train['saturated_vapour_density'] = train['relative_humidity'] * vapour_density







    # Building surface area

    def build_sur_area(x,h):

        side = x**(0.5)

        return 2*(side*side + side*h + side*h)



    train['building_height'] = train['floor_count_ifnan'] * 3  # each floor 3m high

    train['building_vol'] = train['building_height'] * train['square_feet']



    train['build_sur_area'] = build_sur_area(train['square_feet'], train['building_height'] )



    # evaporation rate

    def evaporated_water(x0, x1, x2, x3, x4, x5, time):

        xs = 0.622*(((x0*x1*0.026325) / x3)-1)  # maxm humidity ratio of saturated air

        x = 0.622 * (((x0 * x2 * 0.026325) / x3) - 1)  # humidity ratio of dry air

        if time=='hour': return x4 * x5 *(xs-x)

        if time=='minute': return (x4 * x5 *(xs-x))/60

        if time=='sec': return (x4 * x5 *(xs-x))/3600



    train['evap_coeff'] = 25+19*train['wind_speed']



    train['evap_per_hour'] = evaporated_water(train['relative_humidity'], train['dew_temperature_k'], train['air_temperature_k'],

                                                     train['sea_level_pressure'], train['evap_coeff'], train['build_sur_area'], time='hour')

    train['evap_per_minute'] = evaporated_water(train['relative_humidity'], train['dew_temperature_k'], train['air_temperature_k'],

                                                     train['sea_level_pressure'], train['evap_coeff'], train['build_sur_area'], time='minute')

    train['evap_per_sec'] = evaporated_water(train['relative_humidity'], train['dew_temperature_k'], train['air_temperature_k'],

                                                     train['sea_level_pressure'], train['evap_coeff'], train['build_sur_area'], time='sec')

    train['condense_per_hour'] = train['evap_per_hour']*train['relative_humidity']*100

    train['condense_per_minute'] = train['evap_per_minute']*train['relative_humidity']*100

    train['condense_per_sec'] = train['evap_per_sec']*train['relative_humidity']*100





    # heat required by building

    def required_by_build(x0, x1, x2):

        air_layer = (x0**(1/3)+1)**3 - x0

        air_mass = air_layer*x1

        L = 2256  # latent heat of vaporization of water = 2256 kj/kg

        return  (L*air_mass) * x2



    train['heat_lost_by_build'] = required_by_build(train['building_vol'], train['air_density'], train['evap_per_sec'])



    # dropping some features

    drop_features = ["timestamp"]



    train.drop(drop_features, axis=1, inplace=True)

    # train.dropna(axis=0, how='any')

    train['meter_reading'] = train['meter_reading'].apply(lambda x:np.log1p(x))

    train['meter_reading'] = train['meter_reading'].apply(lambda x:round(x, 2))



    return train

    # if test:

    #     row_ids = train.row_id

    #     train.drop("row_id", axis=1, inplace=True)

    #     return train, row_ids

    # else:

    #     # train.drop("meter_reading", axis=1, inplace=True)

    #     return train, y_train
# reducing memory usage

train = reduce_mem_usage(train_df, use_float16=True)

b_meta = reduce_mem_usage(building_meta_df, use_float16=True)

w_train = reduce_mem_usage(weather_train_df, use_float16=True)



# merging b_meta and w_train to train

train = train.merge(b_meta, on="building_id", how="left")

train = train.merge(w_train, on=["site_id", "timestamp"], how="left")



k_slice=1   # upto the just four slice



def get_train(tr_X, k_slice=1):

    part = len(tr_X)//8

    if k_slice==1: return tr_X[:part]

    elif k_slice==2: return tr_X[part:part*2]

    elif k_slice==3: return tr_X[part*2:part*3]

    elif k_slice==4: return tr_X[part*3:part*4]

    elif k_slice==5: return tr_X[part*4:part*5]

    elif k_slice==6: return tr_X[part*5:part*6]

    elif k_slice==7: return tr_X[part*6:part*7]

    else:return tr_X[part*7:part*8]

train = get_train(train, k_slice=1)

train = preprocess(train)
print(train.shape)

train.head()
del train_df, weather_train_df, building_meta_df

gc.collect()
import h2o

from h2o.automl import H2OAutoML



h2o.init()



x_train = h2o.H2OFrame(train)



x = x_train.columns

y = "meter_reading"

x.remove(y)



# For binary classification, response should be a factor

# train[y] = x_train[y].asfactor()

# test[y] = test[y].asfactor()

# Run AutoML for 20 base models (limited to 1 hour max runtime by default)

aml = H2OAutoML(max_models=40, seed=42, include_algos=['GBM'],

               max_runtime_secs=25200)

aml.train(x=x, y=y, training_frame=x_train)

lb = aml.leaderboard

lb.head(rows=lb.nrows)

l_model = aml.leader
path = '/kaggle/working/model_1.zip'

l_model.download_mojo(path)

FileLink('model_1.zip')
# imported_model = h2o.import_mojo(path)

# # new_observations = h2o.import_file(path='new_observations.csv')

# predictions = imported_model.predict(test)

# predictions
df_test = pd.read_csv(path_test)

weather_test = pd.read_feather(root/'weather_test.feather')

building_meta_df = pd.read_feather(root/'building_metadata.feather')



# reducing memory usage

test = reduce_mem_usage(df_test, use_float16=True)

b_meta = reduce_mem_usage(building_meta_df, use_float16=True)

w_train = reduce_mem_usage(weather_test, use_float16=True)



# merging b_meta and w_train to train

test = test.merge(b_meta, on="building_id", how="left")

test = test.merge(w_train, on=["site_id", "timestamp"], how="left")



X_test = preprocess(test)
del df_test, building, weather_test

gc.collect()
pred = np.expm1(int(l_model.predict(X_test)))//8



pred.to_csv("pred.csv", index=False)# del model_half_1

FileLink('pred.csv')

# gc.collect()



# pred += np.expm1(model_half_2.predict(X_test, num_iteration=model_half_2.best_iteration)) / 2

    

# del model_half_2

# gc.collect()
submission = pd.DataFrame({"row_id": row_ids, "meter_reading": np.clip(pred['predict'], 0, a_max=None)})

submission.to_csv("submission.csv", index=False)

FileLink('submission.csv')