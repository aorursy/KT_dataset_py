# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from matplotlib import pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



air_reserve = pd.read_csv('../input/air_reserve.csv')

hpg_reserve = pd.read_csv('../input/hpg_reserve.csv')



air_store_info = pd.read_csv('../input/air_store_info.csv')

hpg_store_info = pd.read_csv('../input/hpg_store_info.csv')



df_train = pd.read_csv('../input/air_visit_data.csv').sort_values('visit_date')

df_test = pd.read_csv('../input/sample_submission.csv')



store_id_relation = pd.read_csv('../input/store_id_relation.csv')



date_info = pd.read_csv('../input/date_info.csv')

# Any results you write to the current directory are saved as output.
air_reserve.head(10)
air_reserve.visit_datetime.dtype
air_reserve.shape
air_reserve.air_store_id.nunique()
a = air_reserve['air_store_id'].value_counts()

a.head()
air_reserve['air_store_id'].value_counts().idxmax()
air_reserve.loc[air_reserve['air_store_id'] == 'air_8093d0b565e9dbdf']
air_store_info.head(10)
fig, ax = plt.subplots()

fig.set_size_inches(11.7, 8.27)

sns.set_style("whitegrid")

ax = sns.countplot(y = air_store_info['air_genre_name'], data = air_store_info)
fig, ax = plt.subplots()

fig.set_size_inches(15.7, 23.27)

sns.set_style("whitegrid")

ax = sns.countplot(y = air_store_info['air_area_name'], data = air_store_info)
#ax = sns.regplot(x = "longitude", y = "latitude", data = air_store_info)



sns.lmplot('latitude', # Horizontal axis

           'longitude', # Vertical axis

           data = air_store_info, # Data source

           fit_reg = False, # Don't fix a regression line

           #hue="z", # Set color

           #scatter_kws={"marker": "D", # Set marker style

                       # "s": 100}

          )
air_store_info.head()
from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt

f,ax = plt.subplots(1, 1, figsize=(15,16))

m = Basemap(projection='aeqd', width=2000000, height=2000000, lat_0=37.5, lon_0=138.2)



m.drawcoastlines()

m.fillcontinents(color='gray', lake_color='white', zorder=1)

m.scatter(hpg_store_info.longitude.values, hpg_store_info.latitude.values, marker='o', color='red', zorder=999, latlon=True)

ax.legend()

plt.show()
from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt

f,ax = plt.subplots(1, 1, figsize=(15,16))

m = Basemap(projection='aeqd', width=2000000, height=2000000, lat_0=37.5, lon_0=138.2)



m.drawcoastlines()

m.fillcontinents(color='gray', lake_color='white', zorder=1)

m.scatter(air_store_info.longitude.values, air_store_info.latitude.values, marker='o', color='blue', zorder=999, latlon=True)

#m.scatter(air.longitude.values, air.latitude.values, marker='x', color='green', zorder=999, latlon=True)

ax.legend()

plt.show()
from mpl_toolkits.basemap import Basemap



# Create a map on which to draw.  We're using a mercator projection, and showing the whole world.

m = Basemap(projection='merc', llcrnrlat=-80, urcrnrlat=80, llcrnrlon=-180, urcrnrlon=180, lat_ts=20, resolution='c')



# Draw coastlines, and the edges of the map.

m.drawcoastlines()

#m.drawmapboundary()



# Convert latitude and longitude to x and y coordinates

x, y = m(list(air_store_info["longitude"].astype(float)), list(air_store_info["latitude"].astype(float)))



# Use matplotlib to draw the points onto the map.

m.scatter(x, y, 1, marker='o', color='red')



# Show the plot.

plt.show()
''' 

import folium



# Get a basic world map.

air_map = folium.Map(location=[15, 0], zoom_start=1)

# Draw markers on the map.

for name, row in air_store_info.iterrows():

    # For some reason, this one airport causes issues with the map.

    #if row["name"] != "South Pole Station":

        air_map.circle_marker(location=[row["latitude"], row["longitude"]], popup=row["air_store_id"])

# Create and show the map.

air_map.create_map('air.html')

air_map

'''
len(air_store_info)
len(hpg_store_info)
hpg_store_info.hpg_store_id.nunique()
hpg_store_info.head(10)
fig, ax = plt.subplots()

fig.set_size_inches(11.7, 8.27)

sns.set_style("whitegrid")

ax = sns.countplot(y = hpg_store_info['hpg_genre_name'], data = hpg_store_info)
fig, ax = plt.subplots()

fig.set_size_inches(15.7, 28.27)

sns.set_style("whitegrid")

ax = sns.countplot(y = hpg_store_info['hpg_area_name'], data = hpg_store_info)
store_id_relation.isnull().values.any()
store_id_relation.head(10)
len(store_id_relation)
store_id_relation.shape
date_info.isnull().values.any()
date_info.head(10)
len(date_info)
date_info.shape
print('The date ranges between {} and {}'.format(min(date_info.calendar_date), max(date_info.calendar_date)))
f, ax = plt.subplots(figsize=(10, 5))

ax = sns.countplot(x = date_info['holiday_flg'], data = date_info) 
f, ax = plt.subplots(figsize=(13, 8))

ax = sns.countplot(x = date_info['day_of_week'], hue =date_info['holiday_flg'], data = date_info) 
df_train.isnull().values.any()
df_train.head(10)
print('The dates in the train file ranges from {} to {}'.format(min(df_train.visit_date), max(df_train.visit_date)))
print('Train data Head : \n', df_train.head())

print('\n')

print('Test data Head : \n', df_test.head())
print('Train shape : ', df_train.shape)

print('Test shape : ', df_test.shape)
df_test['air_store_id'] = df_test['id'].map(lambda x: '_'.join(x.split('_')[:2]))

df_test['visit_date'] = df_test['id'].map(lambda x: str(x).split('_')[2])



df_test.drop('id', axis=1, inplace=True)
print('Number of Unique train Ids in Train set: ', df_train.air_store_id.nunique())

print('Number of Unique train Ids in Test set: ', df_test.air_store_id.nunique())
train_air_list = df_train.air_store_id.unique().tolist()

test_air_list = df_test.air_store_id.unique().tolist()



len(set(train_air_list) & set(test_air_list))
if(set(test_air_list) == (set(train_air_list) & set(test_air_list))):

    print('True')

else:

    print('False')
del train_air_list

del test_air_list
df_train['visit_date'] = pd.to_datetime(df_train['visit_date'])

df_train['dow'] = df_train['visit_date'].dt.dayofweek

df_train['year'] = df_train['visit_date'].dt.year

df_train['month'] = df_train['visit_date'].dt.month

df_train['visit_date'] = df_train['visit_date'].dt.date
df_train.head()
df_test['visit_date'] = pd.to_datetime(df_test['visit_date'])

df_test['dow'] = df_test['visit_date'].dt.dayofweek

df_test['year'] = df_test['visit_date'].dt.year

df_test['month'] = df_test['visit_date'].dt.month

df_test['visit_date'] = df_test['visit_date'].dt.date
df_test.head()
unique_stores = df_test['air_store_id'].unique()

stores = pd.concat([pd.DataFrame({'air_store_id': unique_stores, 'dow': [i]*len(unique_stores)}) for i in range(7)], axis=0, ignore_index=True).reset_index(drop=True)



stores.head()
tmp = df_train.groupby(['air_store_id','dow'], as_index=False)['visitors'].min().rename(columns={'visitors':'min_visitors'})

stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow']) 

tmp = df_train.groupby(['air_store_id','dow'], as_index=False)['visitors'].mean().rename(columns={'visitors':'mean_visitors'})

stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])

tmp = df_train.groupby(['air_store_id','dow'], as_index=False)['visitors'].median().rename(columns={'visitors':'median_visitors'})

stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])

tmp = df_train.groupby(['air_store_id','dow'], as_index=False)['visitors'].max().rename(columns={'visitors':'max_visitors'})

stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])

tmp = df_train.groupby(['air_store_id','dow'], as_index=False)['visitors'].count().rename(columns={'visitors':'count_observations'})
stores.head()
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow']) 

stores.head()
stores = pd.merge(stores, air_store_info, how='left', on=['air_store_id']) 

stores.head()
from sklearn import *

lbl = preprocessing.LabelEncoder()

stores['air_genre_name'] = lbl.fit_transform(stores['air_genre_name'])

stores['air_area_name'] = lbl.fit_transform(stores['air_area_name'])

stores.head()
date_info = pd.read_csv('../input/date_info.csv').rename(columns={'calendar_date':'visit_date'})

date_info.head()
date_info['visit_date'] = pd.to_datetime(date_info['visit_date'])

date_info['day_of_week'] = lbl.fit_transform(date_info['day_of_week'])

date_info['visit_date'] = date_info['visit_date'].dt.date

date_info.head()
train = pd.merge(df_train, date_info, how='left', on=['visit_date']) 

test = pd.merge(df_test, date_info, how='left', on=['visit_date']) 



train = pd.merge(train, stores, how='left', on=['air_store_id','dow']) 

test = pd.merge(test, stores, how='left', on=['air_store_id','dow'])



train.head()
test.head()
air_reserve.head()
store_id_relation.head()
hpg_reserve = pd.merge(hpg_reserve, store_id_relation, how='inner', on=['hpg_store_id'])

hpg_reserve.head()
train = pd.merge(train, hpg_reserve, how='left', on=['air_store_id','visit_date']) 

test = pd.merge(test, hpg_reserve, how='left', on=['air_store_id','visit_date'])



train = pd.merge(train, air_reserve, how='left', on=['air_store_id','visit_date']) 

test = pd.merge(test, air_reserve, how='left', on=['air_store_id','visit_date']) 
train.head()