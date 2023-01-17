# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import dask

import dask.dataframe as dd

from dask_ml.model_selection import train_test_split

import dask_xgboost

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
bld_meta=dd.read_csv('/kaggle/input/ashrae-energy-prediction/building_metadata.csv')

weather_train=dd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_train.csv',parse_dates=['timestamp'])

weather_test=dd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_test.csv',parse_dates=['timestamp'])

train_df=dd.read_csv('/kaggle/input/ashrae-energy-prediction/train.csv', parse_dates=['timestamp'])

test_df=dd.read_csv('/kaggle/input/ashrae-energy-prediction/test.csv',parse_dates=['timestamp'])
def convert_timestamp(df, time_col):

    df['year']=df[time_col].dt.year

    df['day']=df[time_col].dt.day

    df['weekday']=df[time_col].dt.weekday_name

    df['month']=df[time_col].dt.month_name()

    df['hour']=df[time_col].dt.hour

    

    return df
train_df=convert_timestamp(train_df, 'timestamp')

test_df=convert_timestamp(test_df, 'timestamp')



weather_train=convert_timestamp(weather_train, 'timestamp')

weather_test=convert_timestamp(weather_test, 'timestamp')
train_df['meter']=train_df['meter'].mask(train_df['meter']==0, 'eletricity').mask(train_df['meter']==1, 'chilledwater').mask(train_df['meter']==2,'steam').mask(train_df['meter']==3, 'hotwater')

test_df['meter']=test_df['meter'].mask(test_df['meter']==0, 'eletricity').mask(test_df['meter']==1, 'chilledwater').mask(test_df['meter']==2,'steam').mask(test_df['meter']==3, 'hotwater')
(bld_meta.isnull().sum()/len(bld_meta)).compute()
(weather_train.isnull().sum()/len(weather_train)).compute()
bld_meta=bld_meta.drop(columns=['year_built', 'floor_count'], axis=1)

weather_train=weather_train.drop(columns=['cloud_coverage','precip_depth_1_hr'], axis=1)
bld_weather=dd.merge(weather_train, bld_meta, 

                     on='site_id')

bld_cols=['air_temperature',

       'dew_temperature', 'sea_level_pressure',

       'wind_direction', 'wind_speed', 'year','month', 'day', 'weekday',

       'hour', 'building_id', 'primary_use', 'square_feet']



train_cols=['building_id', 'meter', 'meter_reading','year','month','day','hour']

train_merged_df=dd.merge(train_df[train_cols], bld_weather[bld_cols], how='left', 

                         on=['building_id','year','month','day','hour'])
# takes a while to run

train_pd_df=train_merged_df.compute()
num_cols=['dew_temperature','air_temperature', 'sea_level_pressure','hour',

       'wind_direction', 'wind_speed', 'square_feet','meter_reading']
corr_df=train_pd_df[num_cols].corr()
mask = np.zeros_like(corr_df, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr_df, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})

del corr_df
train_pd_df['log_meter']=np.log(train_pd_df['meter_reading']+1)

sns.distplot(train_pd_df['log_meter'])
select_cols=['wind_speed', 'square_feet','log_meter']

df=train_pd_df.sample(frac=0.1, random_state=22) #sample 10% so the notebook can run quickly, especially the pairplot

df_num=df[select_cols]

print(df_num.shape)

print(df_num.columns)
# takes a loong time to run

sns.pairplot(df_num) 

del df_num
# plot the histograms for each primary use group

select_cols=['primary_use','wind_speed', 'square_feet','log_meter']

df2=train_pd_df[select_cols]

bins = np.arange(0,18, 3)

g = sns.FacetGrid(df2, col="primary_use",col_wrap=4, height=4)

g = g.map(plt.hist, "log_meter", bins=bins)
cat_cols=['primary_use','weekday','month','log_meter']

df_cat=df[cat_cols]
sns.catplot(x="log_meter",y="weekday",kind='violin',data=df_cat)
sns.catplot(x="log_meter",y="primary_use",kind='violin',data=df_cat)
sns.catplot(x="log_meter",y="month",kind='violin',data=df_cat)