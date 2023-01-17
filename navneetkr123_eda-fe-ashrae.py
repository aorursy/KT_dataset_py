# This Python 3 environment comes with many helpful analytics libraries installed

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



# Any results you write to the current directory are saved as output.
build_meta = pd.read_csv("../input/ashrae-energy-prediction/building_metadata.csv")

test = pd.read_csv("../input/ashrae-energy-prediction/test.csv")

train_df = pd.read_csv("../input/ashrae-energy-prediction/train.csv")

weat_train = pd.read_csv("../input/ashrae-energy-prediction/weather_train.csv")

weat_test = pd.read_csv("../input/ashrae-energy-prediction/weather_test.csv")
## Function to reduce the DF size

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
print("Shape of building_metadata is: ", build_meta.shape)

print("Shape of train is: ", train_df.shape)

print("Shape of weat_train is: ", weat_train.shape)

print("Shape of test is: ", test.shape)

print("Shape of weat_test is: ", weat_test.shape)
sub = pd.read_csv("../input/ashrae-energy-prediction/sample_submission.csv")

print("Shape of submission is: ", sub.shape)

sub.head()
build_meta.head()
train_df.head()
weat_train.head()
test.head()
weat_test.head()
import gc
build_meta = reduce_mem_usage(build_meta)

train_df = reduce_mem_usage(train_df)

weat_train = reduce_mem_usage(weat_train)

del test, sub, weat_test

gc.collect()
train = pd.merge(train_df, build_meta, on="building_id", how="left")

print("Shape after merging train and build train is:", train.shape)



del train_df

gc.collect()



train = pd.merge(train, weat_train, on=['site_id','timestamp'], how="left")

print("Shape after merging all train data is:", train.shape)



del weat_train

gc.collect()

import seaborn as sns

def bar_plot(feature, df):

    sns.set(style="darkgrid")

    ax = sns.countplot(x=feature , data=df)

    

train.head()
train.isna().sum()
#bar_plot("meter_reading", train)

print("total different value of meter_reading is:", train.meter_reading.value_counts().shape)
bar_plot("site_id", train)
print("total different value of meter is:", train.meter.value_counts())



bar_plot("meter", train)
print("total different value of primary use is:", train.primary_use.value_counts())

bar_plot("primary_use", train)
print("total different value of square feet is:", train.square_feet.value_counts().shape)

bar_plot("square_feet", train)
print("total different value of year built is:", train.year_built.value_counts())

bar_plot("year_built", train)
print("total different value of floor count is:", train.floor_count.value_counts())

bar_plot("floor_count", train)
print("total different value of air temperature is:", train.air_temperature.value_counts().shape)

bar_plot("air_temperature", train)
print("total different value of cloud coverage is:", train.cloud_coverage.value_counts())

bar_plot("cloud_coverage", train)
print("total different value of dew temperature is:", train.dew_temperature.value_counts().shape)

bar_plot("dew_temperature", train)
print("total different value of precip_depth_1_hr is:", train.precip_depth_1_hr.value_counts().shape)

bar_plot("precip_depth_1_hr", train)
print("total different value of sea_level_pressure is:", train.sea_level_pressure.value_counts().shape)

bar_plot("sea_level_pressure", train)
print("total different value of wind_direction is:", train.wind_direction.value_counts().shape)

print(train.wind_direction.value_counts())

bar_plot("wind_direction", train)
print("total different value of wind_speed is:", train.wind_speed.value_counts().shape)

print(train.wind_speed.value_counts())

bar_plot("wind_speed", train)
import datetime



#convert into datetime

train["timestamp"] = pd.to_datetime(train["timestamp"])



#Extarct year, month, weeks etc from timestamp

train["year"] = pd.DatetimeIndex(train["timestamp"]).year

train["month"] = pd.DatetimeIndex(train["timestamp"]).month

train["day"] = pd.DatetimeIndex(train["timestamp"]).day

train["week"] = pd.DatetimeIndex(train["timestamp"]).week

train.head(2)
print("total different value of year is:", train.year.value_counts().shape)

print(train.year.value_counts())

bar_plot("year", train)
train.drop("year", axis=1, inplace=True)
print("total different value of month is:", train.month.value_counts().shape)

#print(train.month.value_counts())

bar_plot("month", train)
print("total different value of day is:", train.day.value_counts().shape)

#print(train.day.value_counts())

bar_plot("day", train)
print("total different value of week is:", train.week.value_counts().shape)

#print(train.week.value_counts())

bar_plot("week", train)
train.sort_values(by='timestamp', inplace=True)
import matplotlib.pyplot as plt





train['timestamp'].plot()

test_df = pd.read_csv("../input/ashrae-energy-prediction/test.csv")

weat_test = pd.read_csv("../input/ashrae-energy-prediction/weather_test.csv")
test_df = reduce_mem_usage(test_df)

weat_test = reduce_mem_usage(weat_test)

test = pd.merge(test_df, build_meta, on="building_id", how="left")

print("Shape after merging test and build test is:", train.shape)



del test_df

gc.collect()



test = pd.merge(test, weat_test, on=['site_id','timestamp'], how="left")

print("Shape after merging all test data is:", train.shape)



del weat_test, build_meta

gc.collect()

#convert into datetime

test["timestamp"] = pd.to_datetime(test["timestamp"])



#Extarct year, month, weeks etc from timestamp

#train["year"] = pd.DatetimeIndex(train["timestamp"]).year

test["month"] = pd.DatetimeIndex(test["timestamp"]).month

test["day"] = pd.DatetimeIndex(test["timestamp"]).day

test["week"] = pd.DatetimeIndex(test["timestamp"]).week

test.head(2)
train.drop(["timestamp"], axis=1, inplace=True)

test.drop(["timestamp"], axis=1, inplace=True)
target = train["meter_reading"]

del train["meter_reading"]
gc.collect()
from category_encoders import *



def target_encoder(feature):

    cat_vectorizer = TargetEncoder().fit(train[feature].astype(str), target)



    train[feature] = cat_vectorizer.transform(train[feature].astype(str))

    test[feature] = cat_vectorizer.transform(test[feature].astype(str))

    print("Done")

    
cols = ["primary_use"]

for i in cols:

    target_encoder(feature = i)
 #cols = ["floor_count", "year_built", "floor_count", "air_temperature ", "cloud_coverage",\

  #     "dew_temperature", "precip_depth_1_hr", "sea_level_pressure", "wind_direction", "ind_speed"]

train.fillna(-99, inplace=True)

test.fillna(-99, inplace=True)
feat_cols = [cols for cols in train.columns]