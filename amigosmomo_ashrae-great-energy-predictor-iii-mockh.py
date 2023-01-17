# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#import numpy as np # linear algebra

#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#df_sample_submission = pd.read_csv("/kaggle/input/ashrae-energy-prediction/sample_submission.csv")

#df_building_metadata = pd.read_csv("/kaggle/input/ashrae-energy-prediction/building_metadata.csv")

#df_weather_train = pd.read_csv("/kaggle/input/ashrae-energy-prediction/weather_train.csv")

#df_weather_test = pd.read_csv("/kaggle/input/ashrae-energy-prediction/weather_test.csv")

#df_train = pd.read_csv("/kaggle/input/ashrae-energy-prediction/train.csv")

#df_test = pd.read_csv("/kaggle/input/ashrae-energy-prediction/test.csv")
%%time

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



df_train = pd.read_feather('/kaggle/input/ashrae-feather-format-for-fast-loading/train.feather')

df_weather_train = pd.read_feather('/kaggle/input/ashrae-feather-format-for-fast-loading/weather_train.feather')

df_test = pd.read_feather('/kaggle/input/ashrae-feather-format-for-fast-loading/test.feather')

df_weather_test = pd.read_feather('/kaggle/input/ashrae-feather-format-for-fast-loading/weather_test.feather')

df_building_metadata = pd.read_feather('/kaggle/input/ashrae-feather-format-for-fast-loading/building_metadata.feather')

df_sample_submission = pd.read_feather('/kaggle/input/ashrae-feather-format-for-fast-loading/sample_submission.feather')
%%time



df_train["timestamp"] = pd.to_datetime(df_train["timestamp"])

df_test["timestamp"] = pd.to_datetime(df_test["timestamp"])



df_train = df_train.assign(hour=df_train.timestamp.dt.hour,

               day=df_train.timestamp.dt.day,

               month=df_train.timestamp.dt.month,

               year=df_train.timestamp.dt.year)



df_test = df_test.assign(hour=df_test.timestamp.dt.hour,

               day=df_test.timestamp.dt.day,

               month=df_test.timestamp.dt.month,

               year=df_test.timestamp.dt.year)



df_train
########################### Helpers

#################################################################################

## -------------------

## Memory Reducer

# :df pandas dataframe to reduce size             # type: pd.DataFrame()

# :verbose                                        # type: bool

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

## -------------------
%%time

########################### Base check

#################################################################################

do_not_convert = ['category','datetime64[ns]','object']

for df in [df_train, df_test, df_building_metadata, df_weather_train, df_weather_test,df_sample_submission]:

    original = df.copy()

    df = reduce_mem_usage(df)



    for col in list(df):

        if df[col].dtype.name not in do_not_convert:

            if (df[col]-original[col]).sum()!=0:

                df[col] = original[col]

                print('Bad transformation', col)
df_train
import random



mylist = []



for i in range(0,3):

    x = random.randint(1,1448)

    mylist.append(x)



print(mylist)
df_train_polt = df_train[df_train['building_id'].isin(mylist)]

df_train_polt 
df_train_polt.sort_index(axis = 1) 

df_train_polt= df_train_polt.reset_index(drop=True)

df_train_polt
%%time

df_1Day =df_train_polt[((df_train_polt.month == 6) & (df_train_polt.day == 30))]

import seaborn as sns

sns.set(rc={'figure.figsize':(11.2,8.27)})

sns.set(style="darkgrid")

ax = sns.lineplot(x="timestamp", y="meter_reading",hue= 'building_id' ,data= df_1Day)
%%time

df_1month =df_train_polt[((df_train_polt.month == 6))]

import seaborn as sns

sns.set(rc={'figure.figsize':(11.2,8.27)})

sns.set(style="darkgrid")

ax = sns.lineplot(x="timestamp", y="meter_reading",hue= 'building_id' ,data= df_1month)
df_train_polt['meter_reading'][((df_train_polt.building_id == 329)&(df_train_polt.hour == 12))]
%%time

df_1year =df_train_polt[((df_train_polt.hour == 12))]



import seaborn as sns

import matplotlib.pyplot as plt



sns.set(style="dark", context="talk")

sns.set(rc={'figure.figsize':(20,8.27)})

f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20,8.27))

sns.barplot(x=df_train_polt['timestamp'][((df_train_polt.building_id == 329)&(df_train_polt.hour == 12))], y=df_train_polt['meter_reading'][((df_train_polt.building_id == 329)&(df_train_polt.hour == 12))], ax=ax1, color='black')

ax1.axhline(0, color="c", clip_on=False)

ax1.set_ylabel("329")



sns.barplot(x=df_train_polt['timestamp'][((df_train_polt.building_id == 358)&(df_train_polt.hour == 12))], y=df_train_polt['meter_reading'][((df_train_polt.building_id == 358)&(df_train_polt.hour == 12))], ax=ax2, color='black')

ax2.axhline(0, color="c", clip_on=False)

ax2.set_ylabel("358")



sns.barplot(x=df_train_polt['timestamp'][((df_train_polt.building_id == 1019)&(df_train_polt.hour == 12))], y=df_train_polt['meter_reading'][((df_train_polt.building_id == 1019)&(df_train_polt.hour == 12))], ax=ax3, color='black')

ax3.axhline(0, color="c", clip_on=False)

ax3.set_ylabel("1019")



sns.despine(bottom=True)

plt.setp(f.axes, yticks=[])

plt.tight_layout(h_pad=2)