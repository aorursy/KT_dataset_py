# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import re

import matplotlib.pyplot as plt

import matplotlib

import seaborn as sbn

import scipy as sp



%matplotlib inline

color = sbn.color_palette()

pd.set_option('expand_frame_repr', False)

pd.options.mode.chained_assignment = None
train_df = pd.read_json(r'../input/train.json')

train_df.head()
test_df = pd.read_json(r'../input/test.json')

test_df.head()
print("Train_shape : {} \n Test_shape : {}".format(train_df.shape, test_df.shape))
int_lvl = train_df['interest_level'].value_counts()

print(round(int_lvl*100/len(train_df),2))
train_df.dtypes
var_num = ['bathrooms','bedrooms','latitude','longitude','price']

for var in var_num:

    print("Variable : {0} \n{1}\n".format(var,sp.stats.describe(train_df[var])))
train_df.corr()
fig = plt.figure(figsize=(14,7))

ax1 = fig.add_subplot(121)

train_df.hist('bathrooms', bins=20,ax=ax1)

ax2 = fig.add_subplot(122)

train_df.hist('bedrooms', bins=8,ax=ax2)

plt.show()
fig = plt.figure(figsize=(14,7))

ax1 = fig.add_subplot(121)

test_df.hist('bathrooms', bins=20,ax=ax1)

ax2 = fig.add_subplot(122)

test_df.hist('bedrooms', bins=8,ax=ax2)

plt.show()
bth_cnt = test_df['bathrooms'].value_counts()



plt.figure(figsize=(8,6))

sbn.barplot(bth_cnt.index, bth_cnt.values,alpha=0.7)

plt.xlabel('Number of bathrooms', fontsize=12)

plt.ylabel('Number of occurences', fontsize=12)

plt.show()
x_id = np.linspace(0,len(train_df),num=len(train_df),endpoint=False)

fig = plt.figure(figsize=(12,10))

fig.add_subplot(111)

plt.scatter(x_id,train_df['price'])
fig = plt.figure(figsize=(12,10))

ax1 = fig.add_subplot(211)

train_df.plot.scatter('bathrooms','price', ax=ax1)

ax2 = fig.add_subplot(212)

train_df.plot.scatter('bedrooms','price', ax=ax2)
price_bed_bath = train_df.groupby(by=['bedrooms','bathrooms'], axis=0, as_index=False)['price'].median()

price_bed_bath.head()
fig = plt.figure(figsize=(12,10))

ax1 = fig.add_subplot(111)

#bed_bath.plot.bar(['bedrooms','bathrooms'],'listing_id',ax=ax1)

price_bed_bath.plot.bar(['bedrooms','bathrooms'],'price',ax=ax1)

plt.show()
fig = plt.figure(figsize=(12,8))

ax1 = fig.add_subplot(111)

#bed_bath.plot.bar(['bedrooms','bathrooms'],'listing_id',ax=ax1)

train_df[train_df['price']<50000].boxplot(column='price',by=['bedrooms'],ax=ax1)

plt.show()
fig = plt.figure(figsize=(12,8))

ax1 = fig.add_subplot(111)

#bed_bath.plot.bar(['bedrooms','bathrooms'],'listing_id',ax=ax1)

train_df[train_df['price']<50000].boxplot(column='price',by=['bathrooms'],ax=ax1)

plt.show()
int_lvl_an = train_df[['interest_level','bedrooms','bathrooms','listing_id']].groupby(['interest_level','bedrooms','bathrooms'],as_index=False).count()

int_lvl_an[int_lvl_an['listing_id'] > 5] #.sort_values(by='listing_id')
mng_smp = train_df.head()['manager_id']

mng_smp
for i in mng_smp:

    mng_test = train_df[train_df['manager_id']==i][['interest_level','manager_id','listing_id']].groupby(by=['interest_level','manager_id']).count()

    print(mng_test)
build_smp = train_df.head()['building_id']

build_smp
for i in build_smp:

    build_test = train_df[train_df['building_id']==i][['interest_level','building_id','listing_id']].groupby(by=['interest_level','building_id']).count()

    print(build_test)
train_df['created'] = pd.to_datetime(train_df['created'])

train_df['date'] = train_df['created'].dt.date

#train_df['year'] = train_df['created'].dt.year

train_df['month'] = train_df['created'].dt.month

train_df['day'] = train_df['created'].dt.day

train_df['hour'] = train_df['created'].dt.hour

train_df['weekday'] = train_df['created'].dt.weekday

train_df['week'] = train_df['created'].dt.week

#train_df['quarter'] = train_df['created'].dt.quarter
date_list = ['month','day','hour','weekday','week']

for i in date_list:

    dt_vc = train_df[i].value_counts()

    print(dt_vc)
date_cnt = train_df.groupby(['date','interest_level'])['date'].count().unstack('interest_level').fillna(0)

date_cnt[['low','medium','high']].plot(kind='bar', stacked=True, figsize=(13,8))
mnth_cnt = train_df.groupby(['month','interest_level'])['month'].count().unstack('interest_level').fillna(0)

mnth_cnt[['low','medium','high']].plot(kind='bar', stacked=True, figsize=(10,6))
day_cnt = train_df.groupby(['day','interest_level'])['day'].count().unstack('interest_level').fillna(0)

day_cnt[['high','medium','low']].plot(kind='bar', stacked=True, figsize=(13,6))
hr_cnt = train_df.groupby(['hour','interest_level'])['hour'].count().unstack('interest_level').fillna(0)

hr_cnt[['low','medium','high']].plot(kind='bar', stacked=True, figsize=(13,6))
wd_cnt = train_df.groupby(['weekday','interest_level'])['weekday'].count().unstack('interest_level').fillna(0)

wd_cnt[['high','medium','low']].plot(kind='bar', stacked=True, figsize=(13,6))
wk_cnt = train_df.groupby(['week','interest_level'])['week'].count().unstack('interest_level').fillna(0)

wk_cnt[['low','medium','high']].plot(kind='bar', stacked=True, figsize=(13,6))
tr_mn_li = train_df['manager_id'].unique()

ts_mn_li = test_df['manager_id'].unique()

print("Manager_length : \nTrain : {0} \nTest : {1}".format(len(tr_mn_li),len(ts_mn_li)))
tr_bld_li = train_df['building_id'].unique()

ts_bld_li = test_df['building_id'].unique()

print("Building_length : \nTrain : {0} \nTest : {1}".format(len(tr_bld_li),len(ts_bld_li)))
train_df['num_features'] = train_df['features'].apply(lambda x : len(x))

train_df.head()
feature_cnt = train_df.groupby(['num_features','interest_level'])['num_features'].count().unstack('interest_level').fillna(0)

feature_cnt[['low','medium','high']].plot(kind='bar', stacked=True, figsize=(13,6))
train_df['num_photos'] = train_df['photos'].apply(lambda x : len(x))

photos_cnt = train_df.groupby(['num_photos','interest_level'])['num_photos'].count().unstack('interest_level').fillna(0)

photos_cnt[['low','medium','high']].plot(kind='bar', stacked=True, figsize=(13,6))
min_lat = train_df['latitude'].min()

max_lat = train_df['latitude'].max()

min_lon = train_df['longitude'].min()

max_lon = train_df['longitude'].max()



fig = plt.figure(figsize=(12,8))

ax1 = fig.add_subplot(111)

train_df.plot.scatter('latitude','longitude',ax=ax1)

plt.xlim(min_lat,max_lat)

plt.ylim(min_lon, max_lon)

plt.show()
lat_lon = train_df.groupby(['latitude','longitude','interest_level'])['latitude'].count().unstack('interest_level').fillna(0)

lat_lon.head()
lat_lon.shape
lat_lon_test = test_df.groupby(['latitude','longitude'])['latitude'].count().fillna(0)

lat_lon_test.shape
train_df[(train_df['latitude']==0) & (train_df['longitude']==0)]
fig = plt.figure(figsize=(12,8))

ax1 = fig.add_subplot(111)

train_df.plot.scatter('latitude','longitude',ax=ax1)

plt.xlim(30,50)

plt.ylim(-130, -60)

plt.show()
" ".join(i for i in train_df['street_address'].iloc[58].split(" ")[1:3])
len(test_df['street_address'].value_counts())
len(train_df['description'].iloc[893])
#train_df['description_len'] = train_df['description'].apply(lambda x : len(x))

#train_df[train_df['description_len']>=100]['description_len'].count()

#train_df.boxplot(column='description_len', by='interest_level',figsize=(12,8))
def grp(x):

    lx = len(x)

    if lx <= 150:

        return 0

    elif (lx > 150 and lx<=1500):

        return 1

    else:

        return 2
train_df['description_cat'] = train_df['description'].map(grp)
train_df['description_cat'].value_counts()
desc_cnt = train_df.groupby(['description_cat','interest_level'])['description_cat'].count().unstack('interest_level').fillna(0)

desc_cnt[['low','medium','high']].plot(kind='bar', stacked=True, figsize=(10,6))
lat_ulimit = np.percentile(train_df['latitude'],99)

lat_llimit = np.percentile(train_df['latitude'],1)

print(lat_ulimit, lat_llimit)

train_df['latitude'].ix[train_df['latitude']<lat_llimit] = lat_llimit

train_df['latitude'].ix[train_df['latitude']>lat_ulimit] = lat_ulimit
lon_ulimit = np.percentile(train_df['longitude'],99)

lon_llimit = np.percentile(train_df['longitude'],1)

print(lon_ulimit, lon_llimit)

train_df['longitude'].ix[train_df['longitude']<lon_llimit] = lon_llimit

train_df['longitude'].ix[train_df['longitude']>lon_ulimit] = lon_ulimit
train_df['lat_lon'] = train_df['latitude'] + train_df['longitude']

x1 = np.linspace(0,len(train_df),len(train_df),endpoint=False)

plt.scatter(x = x1,y=train_df['lat_lon'])