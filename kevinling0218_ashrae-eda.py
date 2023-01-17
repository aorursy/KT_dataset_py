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
# Import library 

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold

from tqdm import tqdm

import gc

import lightgbm as lgb
# Read in data

df_train = pd.read_csv('../input/ashrae-energy-prediction/train.csv')
df_train.head()
df_train.info()
print ("The number of unique building:", df_train.building_id.nunique())

print ("The number of building_id range:", df_train.building_id.min(), "-", df_train.building_id.max())
df_train.timestamp = pd.to_datetime(df_train.timestamp)
print ("The number of timestamp range:", df_train.timestamp.min(), "-", df_train.timestamp.max())
df_test = pd.read_csv('../input/ashrae-energy-prediction/test.csv')
df_test.head()
df_test.info()
print ("The number of unique building in test:", df_test.building_id.nunique())

print ("The number of building_id range:", df_test.building_id.min(), "-", df_test.building_id.max())
print ("The number of timestamp range in test:", df_test.timestamp.min(), "-", df_test.timestamp.max())
# Read in data

df_meta = pd.read_csv('../input/ashrae-energy-prediction/building_metadata.csv')

df_meta.head()
df_meta.info()
df_w_train = pd.read_csv('../input/ashrae-energy-prediction/weather_train.csv')

df_w_test = pd.read_csv('../input/ashrae-energy-prediction/weather_test.csv')
df_w_train.head()

df_w_test.head()

df_train = df_train.merge(df_meta, on='building_id', how='left')

df_w_train.timestamp = pd.to_datetime(df_w_train.timestamp)

df_train = df_train.merge(df_w_train, on=['site_id', 'timestamp'], how='left')
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
df_train = reduce_mem_usage(df_train)

df_test = reduce_mem_usage(df_test)
# Separate dataframe into different section based on the 

df_elec = df_train[df_train['meter'] == 0]

df_chill = df_train[df_train['meter'] == 1]

df_steam = df_train[df_train['meter'] == 2]

df_hot = df_train[df_train['meter'] == 3]
# Extract the date

df_elec['date'] = df_elec['timestamp'].dt.date

df_chill['date'] = df_chill['timestamp'].dt.date

df_steam['date'] = df_steam['timestamp'].dt.date

df_hot['date'] = df_hot['timestamp'].dt.date
# Extract the hour

df_elec['hour'] = df_elec['timestamp'].dt.hour

df_chill['hour'] = df_chill['timestamp'].dt.hour

df_steam['hour'] = df_steam['timestamp'].dt.hour

df_hot['hour'] = df_hot['timestamp'].dt.hour
# Obtain the mean value at each date

df_elec_date_groupby = df_elec.groupby('date').agg({'meter_reading':'mean'}).reset_index()

df_chill_date_groupby = df_chill.groupby('date').agg({'meter_reading':'mean'}).reset_index()

df_steam_date_groupby = df_steam.groupby('date').agg({'meter_reading':'mean'}).reset_index()

df_hot_date_groupby = df_hot.groupby('date').agg({'meter_reading':'mean'}).reset_index()
# Obtain the mean value at each hour

df_elec_hour_groupby = df_elec.groupby('hour').agg({'meter_reading':'mean'}).reset_index()

df_chill_hour_groupby = df_chill.groupby('hour').agg({'meter_reading':'mean'}).reset_index()

df_steam_hour_groupby = df_steam.groupby('hour').agg({'meter_reading':'mean'}).reset_index()

df_hot_hour_groupby = df_hot.groupby('hour').agg({'meter_reading':'mean'}).reset_index()
f,axes = plt.subplots(2,2, figsize = (20,20))

ax11 = sns.lineplot(x = 'date', y = 'meter_reading', data = df_elec_date_groupby, ax = axes[0][0]).set_title('Electricity')

ax12 = sns.lineplot(x = 'date', y = 'meter_reading', data = df_chill_date_groupby, ax = axes[1][0]).set_title('Chill Water')

ax13 = sns.lineplot(x = 'date', y = 'meter_reading', data = df_steam_date_groupby, ax = axes[0][1]).set_title('Steam')

ax14 = sns.lineplot(x = 'date', y = 'meter_reading', data = df_hot_date_groupby, ax = axes[1][1]).set_title('Hot Water')
f,axes = plt.subplots(2,2, figsize = (20,20))

ax21 = sns.lineplot(x = 'hour', y = 'meter_reading', data = df_elec_hour_groupby, ax = axes[0][0]).set_title('Electricity')

ax22 = sns.lineplot(x = 'hour', y = 'meter_reading', data = df_chill_hour_groupby, ax = axes[1][0]).set_title('Chill Water')

ax23 = sns.lineplot(x = 'hour', y = 'meter_reading', data = df_steam_hour_groupby, ax = axes[0][1]).set_title('Steam')

ax24 = sns.lineplot(x = 'hour', y = 'meter_reading', data = df_hot_hour_groupby, ax = axes[1][1]).set_title('Hot Water')
df_steam_edu = df_steam[df_steam['primary_use'] =='Education']
ax = sns.lineplot(x = 'date', y = 'meter_reading', data = df_steam_edu).set_title('Steam for Education')
df_steam_office = df_steam[df_steam['primary_use'] =='Office']
ax = sns.lineplot(x = 'date', y = 'meter_reading', data = df_steam_office).set_title('Steam for Office Building')
df_steam_residential = df_steam[df_steam['primary_use'] =='Lodging/residential']

ax = sns.lineplot(x = 'date', y = 'meter_reading', data = df_steam_residential).set_title('Steam for Residential Building')
df_chill_edu = df_chill[df_chill['primary_use'] =='Education']

df_chill_office = df_chill[df_chill['primary_use'] =='Office']

df_chill_residential = df_chill[df_chill['primary_use'] =='Lodging/residential']

df_chill_public = df_chill[df_chill['primary_use'] =='Public services']

f,axes = plt.subplots(2,2, figsize = (20,20))

ax21 = sns.lineplot(x = 'date', y = 'meter_reading', data = df_chill_edu, ax = axes[0][0]).set_title('Chill - Education')

ax22 = sns.lineplot(x = 'date', y = 'meter_reading', data = df_chill_office, ax = axes[1][0]).set_title('Chill - Office')

ax23 = sns.lineplot(x = 'date', y = 'meter_reading', data = df_chill_residential, ax = axes[0][1]).set_title('Chill - Residential')

ax24 = sns.lineplot(x = 'date', y = 'meter_reading', data = df_chill_public, ax = axes[1][1]).set_title('Chill - Public')
#Group by site_id 

df_elec_gb = df_elec.groupby(['site_id', 'hour'])['meter_reading'].mean().reset_index()
df_elec_gb_site1 = df_elec_gb[df_elec_gb['site_id'] == 1]

df_elec_gb_site2 = df_elec_gb[df_elec_gb['site_id'] == 2]

df_elec_gb_site3 = df_elec_gb[df_elec_gb['site_id'] == 3]

df_elec_gb_site4 = df_elec_gb[df_elec_gb['site_id'] == 4]

df_elec_gb_site5 = df_elec_gb[df_elec_gb['site_id'] == 5]

df_elec_gb_site6 = df_elec_gb[df_elec_gb['site_id'] == 6]

df_elec_gb_site7 = df_elec_gb[df_elec_gb['site_id'] == 7]

df_elec_gb_site8 = df_elec_gb[df_elec_gb['site_id'] == 8]

f,axes = plt.subplots(4,2, figsize = (20,20))

ax21 = sns.lineplot(x = 'hour', y = 'meter_reading', data = df_elec_gb_site1, ax = axes[0][0]).set_title('Site1')

ax22 = sns.lineplot(x = 'hour', y = 'meter_reading', data = df_elec_gb_site2, ax = axes[1][0]).set_title('Site2')

ax23 = sns.lineplot(x = 'hour', y = 'meter_reading', data = df_elec_gb_site3, ax = axes[0][1]).set_title('Site3')

ax24 = sns.lineplot(x = 'hour', y = 'meter_reading', data = df_elec_gb_site4, ax = axes[1][1]).set_title('Site4')

ax21 = sns.lineplot(x = 'hour', y = 'meter_reading', data = df_elec_gb_site5, ax = axes[2][0]).set_title('Site5')

ax22 = sns.lineplot(x = 'hour', y = 'meter_reading', data = df_elec_gb_site6, ax = axes[3][0]).set_title('Site6')

ax23 = sns.lineplot(x = 'hour', y = 'meter_reading', data = df_elec_gb_site7, ax = axes[2][1]).set_title('Site7')

ax24 = sns.lineplot(x = 'hour', y = 'meter_reading', data = df_elec_gb_site8, ax = axes[3][1]).set_title('Site8')