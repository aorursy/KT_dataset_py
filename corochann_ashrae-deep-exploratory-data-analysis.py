import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

import gc



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn import preprocessing

import seaborn as sns

import random

from sklearn.model_selection import KFold

import lightgbm as lgb

import gc

import tqdm

import lightgbm as lgb

import xgboost as xgb

import catboost as cb

# Read data...

root = '../input/ashrae-energy-prediction'



train_df = pd.read_csv(os.path.join(root, 'train.csv'))

weather_train_df = pd.read_csv(os.path.join(root, 'weather_train.csv'))

test_df = pd.read_csv(os.path.join(root, 'test.csv'))

weather_test_df = pd.read_csv(os.path.join(root, 'weather_test.csv'))

building_meta_df = pd.read_csv(os.path.join(root, 'building_metadata.csv'))

sample_submission = pd.read_csv(os.path.join(root, 'sample_submission.csv'))
train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])

test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])

weather_train_df['timestamp'] = pd.to_datetime(weather_train_df['timestamp'])

weather_test_df['timestamp'] = pd.to_datetime(weather_test_df['timestamp'])
print('train_df shape: ', train_df.shape)

train_df.sample(3)
train_df.iloc[[0,-1]]
print('test_df shape: ', test_df.shape)

test_df.sample(3)
test_df.sort_values(by=['timestamp']).iloc[[0, -1]]
print(f"number of building_id in train {len(train_df['building_id'].unique())}, test {len(test_df['building_id'].unique())}")
print('sample_submission.shape', sample_submission.shape)

sample_submission.sample(3)
print('building_meta_df shape', building_meta_df.shape)

building_meta_df.sample(3)
primary_use_list = building_meta_df['primary_use'].unique()

primary_use_list
building_meta_df['site_id'].unique()
print('weather_train_df shape', weather_train_df.shape)

weather_train_df.sample(3)
print('weather_test_df shape', weather_test_df.shape)

weather_test_df.sample(3)
fig, ax = plt.subplots()



test_df.meter.hist(ax=ax, color=[0., 0., 1., 0.5])

train_df.meter.hist(ax=ax, color=[1., 0., 0., 0.5])



# {0: electricity, 1: chilledwater, 2: steam, hotwater: 3}

ax.set_xticks(np.arange(4))

ax.set_xticklabels(['electricity', 'chilledwater', 'steam', 'hotwater'])

plt.show()
sns.distplot(train_df['meter_reading']).set_title('meter_reading', fontsize=16)

plt.show()

sns.distplot(np.log1p(train_df['meter_reading'])).set_title('meter_reading', fontsize=16)

plt.show()
train_df['meter_reading_log1p'] = np.log1p(train_df['meter_reading'])



titles = ['electricity', 'chilledwater', 'steam', 'hotwater']



fig, axes = plt.subplots(1, 4, figsize=(18, 5))

for i in range(4):

    title = titles[i]

    sns.distplot(train_df.loc[train_df['meter'] == i, 'meter_reading_log1p'], ax=axes[i]).set_title(title, fontsize=16)



plt.show()
# # takes too much time...

# train_df['meter_reading_log1p'] = np.log1p(train_df['meter_reading'])

# sns.violinplot(x='meter', y='meter_reading_log1p', data=train_df)
unique_time = train_df['timestamp'].unique()

unique_time[:25], unique_time[-25:]
print('train_df: total rows', len(train_df))

print('Number of nan')

train_df.isna().sum()
print('building_meta_df: total rows', len(building_meta_df))

print('Number of nan')

building_meta_df.isna().sum()
print('weather_train_df: total rows', len(weather_train_df))

print('Number of nan')

weather_train_df.isna().sum()
# only look specific building

target_building_id = 500

target_meter = 0



def plot_meter_reading_in_time(target_building_id, target_meter, target_month):

    target_building_df = train_df[(train_df['building_id'] == target_building_id) & (train_df['meter'] == target_meter)].copy()

    target_building_df['hour'] = target_building_df.timestamp.dt.hour

    target_building_df['month'] = target_building_df.timestamp.dt.month

    target_building_df['day'] = target_building_df.timestamp.dt.day

    target_building_df['dow'] = target_building_df.timestamp.dt.dayofweek



    plt.figure()

    plt.title(f'building_id {target_building_id} meter {target_meter} month {target_month}')

    for day in range(1, 8):

        target_building_df_short = target_building_df[(target_building_df['month'] == target_month) & (target_building_df['day'] == day)]

        plt.plot(target_building_df_short['hour'].values, target_building_df_short['meter_reading_log1p'].values, label=f'day {day}: dow {target_building_df_short.dow.values[0]}')

    plt.legend()

    # plt.scatter(target_building_df['hour'].values, target_building_df['meter_reading_log1p'].values)
plot_meter_reading_in_time(target_building_id=300, target_meter=0, target_month=4)

plot_meter_reading_in_time(target_building_id=500, target_meter=0, target_month=6)

plot_meter_reading_in_time(target_building_id=900, target_meter=0, target_month=9)
for target_month in range(1, 13, 1):

    plot_meter_reading_in_time(target_building_id=112, target_meter=3, target_month=target_month)

# categorize primary_use column to reduce memory on merge...



primary_use_dict = {key: value for value, key in enumerate(primary_use_list)} 

print('primary_use_dict: ', primary_use_dict)

building_meta_df['primary_use'] = building_meta_df['primary_use'].map(primary_use_dict)



gc.collect()
from pandas.api.types import is_datetime64_any_dtype as is_datetime



# copy from https://www.kaggle.com/ryches/simple-lgbm-solution

#Based on this great kernel https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65

def reduce_mem_usage(df):

    start_mem_usg = df.memory_usage().sum() / 1024**2 

    print("Memory usage of properties dataframe is :",start_mem_usg," MB")

    NAlist = [] # Keeps track of columns that have missing values filled in. 

    for col in df.columns:

        col_type = df[col].dtype

        if col_type != object and not is_datetime(df[col]):  # Exclude strings            

            # Print current column type

            print("******************************")

            print("Column: ",col)

            print("dtype before: ",df[col].dtype)            

            # make variables for Int, max and min

            IsInt = False

            mx = df[col].max()

            mn = df[col].min()

            print("min for this col: ",mn)

            print("max for this col: ",mx)

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

            print("******************************")

    # Print final result

    print("___MEMORY USAGE AFTER COMPLETION:___")

    mem_usg = df.memory_usage().sum() / 1024**2 

    print("Memory usage is: ",mem_usg," MB")

    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")

    return df, NAlist
reduce_mem_usage(train_df)

reduce_mem_usage(test_df)

reduce_mem_usage(building_meta_df)

reduce_mem_usage(weather_train_df)

reduce_mem_usage(weather_test_df)
gc.collect()
# merge building and weather information to test data...

# be careful that it takes a lot of memory



test_df = test_df.merge(building_meta_df, on='building_id', how='left')

test_df = test_df.merge(weather_test_df, on=['site_id', 'timestamp'], how='left')

del weather_test_df

gc.collect()
# merge building and weather information to train data...



train_df = train_df.merge(building_meta_df, on='building_id', how='left')

train_df = train_df.merge(weather_train_df, on=['site_id', 'timestamp'], how='left')



del building_meta_df

del weather_train_df

gc.collect()
train_df.head()
test_df.head()
gc.collect()
train_df.to_feather('train.feather')

test_df.to_feather('test.feather')

sample_submission.to_feather('sample_submission.feather')
train_df.shape