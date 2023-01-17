# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("/kaggle/input/ashrae-energy-prediction/train.csv")

test = pd.read_csv("/kaggle/input/ashrae-energy-prediction/test.csv")

weather_train = pd.read_csv("/kaggle/input/ashrae-energy-prediction/weather_train.csv")

weather_test = pd.read_csv("/kaggle/input/ashrae-energy-prediction/weather_test.csv")

building_data = pd.read_csv("/kaggle/input/ashrae-energy-prediction/building_metadata.csv")
# taken from https://www.kaggle.com/caesarlupum/ashrae-start-here-a-gentle-introduction



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
train = reduce_mem_usage(train)

test = reduce_mem_usage(test)



weather_train = reduce_mem_usage(weather_train)

weather_test = reduce_mem_usage(weather_test)

building_data = reduce_mem_usage(building_data)
train
weather_train
building_data
building_data["primary_use"].value_counts()
train = train.merge(building_data, left_on = "building_id", right_on = "building_id", how = "left")
train = train.merge(weather_train, left_on = ["site_id", "timestamp"], right_on = ["site_id", "timestamp"], how = "left")
train
edu_1 = train.loc[train['building_id'] == 1]

ent_10 = train.loc[train['building_id'] == 10]

food_179 = train.loc[train['building_id'] == 179]

health_208 = train.loc[train['building_id'] == 208]

lodg_6 = train.loc[train['building_id'] == 16]

inds_672 = train.loc[train['building_id'] == 672]

office_9 = train.loc[train['building_id'] == 9]

other_42 = train.loc[train['building_id'] == 42]

parking_51 = train.loc[train['building_id'] == 51]

public_138 = train.loc[train['building_id'] == 138]

relig_186 = train.loc[train['building_id'] == 186]

serv_892 = train.loc[train['building_id'] == 892]

tech_575 = train.loc[train['building_id'] == 575]

util_285 = train.loc[train['building_id'] == 285]

storage_164 = train.loc[train['building_id'] == 164]
edu_1.plot.line(x='timestamp', y='meter_reading',figsize=(20,5), title="Education")

ent_10.plot.line(x='timestamp', y='meter_reading',figsize=(20,5), title="Entertainment")

food_179.plot.line(x='timestamp', y='meter_reading',figsize=(20,5), title="Food")

health_208.plot.line(x='timestamp', y='meter_reading',figsize=(20,5), title="Healthcare")

lodg_6.plot.line(x='timestamp', y='meter_reading',figsize=(20,5), title="Lodging")

inds_672.plot.line(x='timestamp', y='meter_reading',figsize=(20,5), title="Industrial")

office_9.plot.line(x='timestamp', y='meter_reading',figsize=(20,5), title="Office")

other_42.plot.line(x='timestamp', y='meter_reading',figsize=(20,5), title="Other")

parking_51.plot.line(x='timestamp', y='meter_reading',figsize=(20,5), title="Parking")

public_138.plot.line(x='timestamp', y='meter_reading',figsize=(20,5), title="Public")

relig_186.plot.line(x='timestamp', y='meter_reading',figsize=(20,5), title="Religious")

serv_892.plot.line(x='timestamp', y='meter_reading',figsize=(20,5), title="Services")

tech_575.plot.line(x='timestamp', y='meter_reading',figsize=(20,5), title="Technology")

util_285.plot.line(x='timestamp', y='meter_reading',figsize=(20,5), title="Utility")

storage_164.plot.line(x='timestamp', y='meter_reading',figsize=(20,5), title="Storage")
for i in range(16):

    sns.catplot(x="primary_use", y="meter_reading", data=train.loc[train['site_id'] == i], height=10, aspect=2)
train.groupby("primary_use")["meter_reading"].mean()
train["month"] = train["timestamp"].apply(lambda x : x.split(" ")[0].split("-")[1])

train["day"] = train["timestamp"].apply(lambda x : x.split(" ")[0].split("-")[2])

train["time"] = train["timestamp"].apply(lambda x : x.split(" ")[1].split(":")[0])

train["day_of_week"] = pd.DatetimeIndex(train["timestamp"].apply(lambda x : x.split(" ")[0])).dayofweek

train
for i in range(10):

    sns.catplot(x="day_of_week", y="meter_reading", hue="primary_use", data=train.loc[train['site_id'] == i], height=10, aspect=2)
missing = train.isnull().sum() * 100 / len(train)

missing_df = pd.DataFrame({'column_name': train.columns,

                                 'percent_missing': missing})

missing_df.sort_values('percent_missing', inplace=True)

missing_df