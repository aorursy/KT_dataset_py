# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/data-cleaning-metadata-and-irrigation'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
os.chdir('/kaggle/input/data-cleaning-metadata-and-irrigation/')

meta = pd.read_csv('metadata_kaggle_cleaned.csv')

print(meta.shape)

meta.head()
eda = meta.copy()

eda.info()
target = meta['rating'].replace(['C1', 'D1'], ['C', 'D'])

target.unique()
train = eda[meta['rating'].notna()]

print(train.shape)

train.head()
train.info()
pd.value_counts(target).plot.bar(figsize=(15,5), title='Distribution of Target Variable (rating)')
eda['rating'] = eda['rating'].replace('C1', 'C')

eda['rating'] = eda['rating'].replace('D1', 'D')

eda['rating'].unique()
train['rating'] = train['rating'].replace('C1', 'C')

train['rating'] = train['rating'].replace('D1', 'D')

train['rating'].unique()
pd.value_counts(train['rating']).plot.bar(figsize=(15,5), title='Distriution of Target Variable (rating)')
fig, axs = plt.subplots(2,4, figsize=(20,8))

pd.value_counts(meta['irrigation']).plot.bar(ax=axs[0, 0], rot=0, subplots=True)

pd.value_counts(meta['steam']).plot.bar(ax=axs[0, 1], rot=0, subplots=True)

pd.value_counts(meta['chilledwater']).plot.bar(ax=axs[0, 2], rot=0, subplots=True)

pd.value_counts(meta['hotwater']).plot.bar(ax=axs[0, 3], rot=0, subplots=True)

pd.value_counts(meta['electricity']).plot.bar(ax=axs[1, 0], rot=0, subplots=True)

pd.value_counts(meta['water']).plot.bar(ax=axs[1, 1], rot=0, subplots=True)

pd.value_counts(meta['solar']).plot.bar(ax=axs[1, 2], rot=0, subplots=True)

pd.value_counts(meta['gas']).plot.bar(ax=axs[1, 3], rot=0, subplots=True)
meta.loc[:, 'yearbuilt':'Steam Heating'].isin([0]).sum().sort_values(ascending=False)
fig, axs = plt.subplots(2,4, figsize=(20,8))

pd.value_counts(train['irrigation']).plot.bar(ax=axs[0, 0], rot=0, subplots=True)

pd.value_counts(train['steam']).plot.bar(ax=axs[0, 1], rot=0, subplots=True)

pd.value_counts(train['chilledwater']).plot.bar(ax=axs[0, 2], rot=0, subplots=True)

pd.value_counts(train['hotwater']).plot.bar(ax=axs[0, 3], rot=0, subplots=True)

pd.value_counts(train['electricity']).plot.bar(ax=axs[1, 0], rot=0, subplots=True)

pd.value_counts(train['water']).plot.bar(ax=axs[1, 1], rot=0, subplots=True)

pd.value_counts(train['solar']).plot.bar(ax=axs[1, 2], rot=0, subplots=True)

pd.value_counts(train['gas']).plot.bar(ax=axs[1, 3], rot=0, subplots=True)
train.loc[:, 'yearbuilt':'Steam Heating'].isin([0]).sum().sort_values(ascending=False)
train = train.drop(['solar', 'chilledwater', 'water', 'steam', 'irrigation'], axis=1)

eda = eda.drop(['solar', 'chilledwater', 'water', 'steam', 'irrigation'], axis=1)

eda.info()
drop_cols = ['eui', 'leed_level', 'Oil Heating', 'Boiler fed central heating', 'Gas Boilers', 'District Heating',

             'Heat network but not ours', 'Heat network and steam', 'Steam Heating', 'Electric Heating',

             'Electricity', 'Heat network']



eda = eda.drop(drop_cols, axis=1)

train = train.drop(drop_cols, axis=1)

eda.info()
# dealing with the nulls in yearbuilt in the training data by filling in with mean of entire dataset

eda['yearbuilt'] = eda['yearbuilt'].fillna(eda.yearbuilt.mean()).astype('int64')

train['yearbuilt'] = train['yearbuilt'].fillna(eda.yearbuilt.mean()).astype('int64')

train.info()
pd.value_counts(train['yearbuilt']).plot.bar(title='distribution of year built', figsize=(10,5))
fig, axs = plt.subplots(1,2, figsize=(20,5))

pd.value_counts(train['usage']).plot.bar(title='Distribution of Usage in Train', ax=axs[0], rot=0)

pd.value_counts(train['subusage']).plot.bar(title='Distribution of Subusage in Train', ax=axs[1])
u1 = train[train['usage']=='Education']

u2 = train[train['usage']=='Government']

u = pd.concat([u1, u2])

usages = pd.get_dummies(u['usage'])

train = pd.concat([u, usages], axis=1).drop('usage', axis=1)

train
train = train.drop('subusage', axis=1)

eda = eda.drop('subusage', axis=1)

eda
import seaborn as sns

fig, axs = plt.subplots(1,2, figsize=(20,5))

sns.distplot(train['sqm'], ax=axs[0]).set_title('Distribution of Building sqm')

sns.distplot(train['sqft'], ax=axs[1]).set_title('Distribution of Building sqft')
fig, axs = plt.subplots(1,2, figsize=(20,5))

pd.value_counts(train['timezone']).plot.bar(title='Distribution of Time Zones in Train', ax=axs[0])

pd.value_counts(eda['timezone']).plot.bar(title='Distribution of Time Zones in Full Dataset', ax=axs[1])
eur1 = train[train['timezone']=='Europe/London']

eur2 = train[train['timezone']=='Europe/Dublin']

eur = pd.concat([eur1, eur2])

timezones = pd.get_dummies(eur['timezone'])

train = pd.concat([eur, timezones], axis=1).drop('timezone', axis=1)

train
eur1 = eda[eda['timezone']=='Europe/London']

eur2 = eda[eda['timezone']=='Europe/Dublin']

eur = pd.concat([eur1, eur2])

timezones = pd.get_dummies(eur['timezone'])

eur_eda = pd.concat([eur, timezones], axis=1).drop('timezone', axis=1)

eur_eda
u1 = eur_eda[eur_eda['usage']=='Education']

u2 = eur_eda[eur_eda['usage']=='Government']

u = pd.concat([u1, u2])

usages = pd.get_dummies(u['usage'])

eur_eda = pd.concat([u, usages], axis=1).drop('usage', axis=1)

eur_eda
fig, axs = plt.subplots(1,2, figsize=(20,5))

plt1 = pd.value_counts(train['site_id_kaggle']).plot.bar(ax=axs[0], title='Distribution of Kaggle Site IDs')

plt2 = pd.value_counts(train['site_id']).plot.bar(ax=axs[1], title='Distribution of Site IDs')
fig, axs = plt.subplots(1,2, figsize=(20,5))

pd.value_counts(train['lat']).plot.bar(ax=axs[0], title='Distribution of Latitudes')

pd.value_counts(train['lng']).plot.bar(ax=axs[1], title='Distribution of Longitudes')
eda = eda.drop(['site_id_kaggle', 'building_id_kaggle'], axis=1)

eur_eda = eur_eda.drop([ 'site_id_kaggle', 'building_id_kaggle'], axis=1)

train = train.drop(['site_id_kaggle', 'building_id_kaggle'], axis=1)

eda.info()
eur_eda.info()
train.info()
ids = eur_eda['building_id']

ids = ids.unique()

ids
def annual(path, data, name):

    name1 = '2016_'+name

    name2 = '2017_'+name

    os.chdir(path)

    d = pd.read_csv(data)

    d['Datetime'] = pd.to_datetime(d['timestamp'])

    d = d.set_index(pd.DatetimeIndex(d['Datetime']))

    d = d.drop('timestamp', axis=1)

    d = d.resample('Y').mean()

    d = d.T.reset_index().rename(columns={'index': 'building_id'})

    d = d.rename_axis('index')

    d = d.rename(columns={d.columns[1]: name1, d.columns[2]: name2})

    return d    
annual_hotWater = annual('/kaggle/input/hot-eda/', 'hot_water_cleaned.csv', 'HotWater')

annual_gas = annual('/kaggle/input/gas-eda', 'gas_cleaned_new.csv', 'Gas')

annual_electricity = annual('/kaggle/input/electricity-data-cleaning', 'electricity_cleaned_new.csv', 'Electricity')
eur_eda = pd.merge(left=eur_eda, right=annual_hotWater, how='left', left_on='building_id', right_on='building_id')

eur_eda = pd.merge(left=eur_eda, right=annual_gas, how='left', left_on='building_id', right_on='building_id')

eur_eda = pd.merge(left=eur_eda, right=annual_electricity, how='left', left_on='building_id', right_on='building_id')



eur_eda[['2016_HotWater', '2016_Gas', '2016_Electricity']] = eur_eda[['2016_HotWater', '2016_Gas', '2016_Electricity']].fillna(-1)

eur_eda[['2017_HotWater', '2017_Gas', '2017_Electricity']] = eur_eda[['2017_HotWater', '2017_Gas', '2017_Electricity']].fillna(-1)

eur_eda.info()
train = eur_eda[eur_eda['rating'].notna()]

train.info()
test = eur_eda[eur_eda['rating'].isnull()]

test = test.drop('rating', axis=1)

test.info()
os.chdir('/kaggle/input/hot-eda/')

hot = pd.read_csv('hot_water_cleaned.csv')

print(hot.shape)

hot.head()
hot.dtypes
hot['Datetime'] = pd.to_datetime(hot['timestamp'])

hot = hot.set_index(pd.DatetimeIndex(hot['Datetime']))

hot = hot.drop('timestamp', axis=1)

hot.head()
hot = hot.resample('Y').mean()

print(hot.shape)

hot.head()
hot = hot.T.reset_index().rename(columns={'index': 'building_id'})

hot = hot.rename_axis('index')

hot.head()
hot.dtypes
hot = hot.rename(columns={hot.columns[1]: '2016_Avg_HotWater', hot.columns[2]: '2017_Avg_HotWater'})

hot.info()
#eur_eda = pd.merge(left=eur_eda, right=hot, how='left', left_on='building_id', right_on='building_id')

#eur_eda['2016_Avg_HotWater'] = eur_eda['2016_Avg_HotWater'].fillna(-1)

#eur_eda['2017_Avg_HotWater'] = eur_eda['2017_Avg_HotWater'].fillna(-1)

eur_eda.info()
#train = eur_eda[eur_eda['rating'].notna()]

train.info()
#test = eur_eda[eur_eda['rating'].isnull()]

#test = test.drop('rating', axis=1)

test.info()
train.to_csv('/kaggle/working/train_rating_eu.csv')

test.to_csv('/kaggle/working/test_rating_eu.csv')