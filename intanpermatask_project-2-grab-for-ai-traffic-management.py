import pandas as pd

import numpy as np

import Geohash

import itertools



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import warnings 

warnings.filterwarnings("ignore", category=DeprecationWarning)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/grabtrafficdata/training.csv")

df.head()
df = df.groupby('geohash6').filter(lambda x : len(x)>1344)
# We begin with generating unique values of geohash and day, and also range of timestamp

geohash6_2 = df['geohash6'].unique()

day_2 = df['day'].unique()

timestamp_2 = pd.date_range("00:00", "23:45", freq="15min").strftime('%H:%M')
# `new_data` is a list containing each unique geohash, day and timestamp

new_data = [geohash6_2, day_2, timestamp_2]



# `data` is a list containing combination of each unique geohash, day and timestamp

data = list(itertools.product(*new_data))



# Next we create a dataframe named `train` that contains all data points in `data`

train = pd.DataFrame(data = data, columns = ['geohash6', 'day', 'timestamp'])



train.head()
# Finally we merge the `demand` data from df to train and replace NaN with 0.

df_main = pd.merge(train, df, on=['geohash6','day', 'timestamp'], how='left')

df_main= df_main.fillna(0)

df_main.head()
df_main.info(memory_usage="deep")
df_main.memory_usage(deep=True) * 1e-6
print("size before:", df_main["geohash6"].memory_usage(deep=True) * 1e-6)

df_main["geohash6"] = df_main["geohash6"].astype("category")

print("size after :", df_main["geohash6"].memory_usage(deep=True) * 1e-6)
print("size before:", df_main["timestamp"].memory_usage(deep=True) * 1e-6)

df_main.timestamp = pd.to_timedelta(df_main.timestamp +':00')

print("size after: ", df_main["timestamp"].memory_usage(deep=True) * 1e-6)
# Let's have a look again on the main_df information

df_main.info(memory_usage="deep")
# And let's check the first 5 rows of our processed dataframe.

df_main.head()
df_main.describe()
df_main['timestamp'].nunique()
df_main['day'].nunique()
df_main['geohash6'].nunique()
# sort day and timestamp

df_main = df_main.sort_values(['day', 'timestamp'], ascending=[True, True])
df_main = df_main.set_index(['day', 'timestamp'])

df_main.tail()
# create train dataframe

df_train = df_main.loc[(slice(1, 50), slice(None)), :]

df_train.tail()
# create validation dataframe

df_val = df_main.loc[(slice(51, 61), slice(None)), :]

df_val.tail()