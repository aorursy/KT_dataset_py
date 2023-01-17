# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

import timeit

import math



# vectorized error calc

def rmsle(y, y0):

    assert len(y) == len(y0)

    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

print(train_df.shape)

print(test_df.shape)
print(train_df['timestamp'].min())

print(train_df['timestamp'].max())

print(test_df['timestamp'].min())

print(test_df['timestamp'].max())
train_df = train_df.sort_values(by='timestamp')



# split train / test

train = train_df[train_df['timestamp']< "2014-08-01"]

test  = train_df[train_df['timestamp']>= "2014-08-01"]
print(train.shape)

print(test.shape)
plt.scatter(train_df['id'], train_df['price_doc'], alpha=0.5, cmap='viridis')



plt.title('Price per transaction, in chronological order')

plt.xlabel('id')

plt.ylabel('price')



plt.ylim(0, 20000000)

plt.show()
train_df.head(1)
# macro economics data

macro_df = pd.read_csv("../input/macro.csv")

macro_df.head(1)
# moving average



# adding macro economic data

train_df = pd.merge(train_df, macro_df, how='left', on=['timestamp'])



# first let's average per day

gb = train_df.groupby(['timestamp'])

gb.sum().head()

dfagg = pd.DataFrame()



dfagg['avg_price_per_sqm'] = gb.price_doc.sum() / gb.full_sq.sum()

dfagg['rolling_average_immo'] = dfagg['avg_price_per_sqm'].rolling(30).mean()



dfagg['oil_avg_price'] = gb.oil_urals.mean()

dfagg['rolling_average_oil'] = dfagg['oil_avg_price'].rolling(30).mean()



dfagg['oil_avg_price_2'] = gb.brent.mean()

dfagg['rolling_average_oil_2'] = dfagg['oil_avg_price_2'].rolling(30).mean()



dfagg.reset_index(inplace=True)

dfagg['date'] = pd.to_datetime(dfagg['timestamp'])



plt.figure(figsize=(14,8))

plt.plot(dfagg['date'], dfagg['rolling_average_immo'], label='avg price per square meter')

plt.plot(dfagg['date'], 1200 * dfagg['rolling_average_oil'], label = 'oil_urals')

plt.plot(dfagg['date'], 1200 * dfagg['rolling_average_oil_2'], label='brent')



plt.title('Rolling average price per square meter')

#plt.xlabel('days')

plt.ylabel('average price per full_sqm')



plt.legend(loc='lower right')



plt.ylim(20000, 180000)

plt.show()
# price depending on the distance to Kremlin

train['kremlin_km_rounded'] = np.round(train['kremlin_km'])

gb = train.groupby(['kremlin_km_rounded'])

dfagg = pd.DataFrame()

dfagg['avg_price_per_kremlin_km'] = gb.price_doc.mean()

dfagg.reset_index(inplace=True)



plt.figure(figsize=(14,8))

plt.scatter(dfagg['kremlin_km_rounded'], dfagg['avg_price_per_kremlin_km'], label='avg price')

plt.title('Rolling average price per square meter')

plt.ylabel('distance to Kremlin')

plt.legend(loc='lower right')

plt.show()
train['date'] = pd.to_datetime(train['timestamp'])

x = train.groupby(['date']).count()

x.reset_index(inplace=True)



plt.figure(figsize=(14,8))

plt.plot(x['date'], x['id'])

plt.title('Number of transactions per day')

plt.show()
x = train.groupby(['date']).mean()

x.reset_index(inplace=True)



plt.figure(figsize=(14,8))

plt.plot(x['date'], x['price_doc'])

plt.title('Average price per day')

plt.show()
train.price_doc.mean()
rmsle(np.repeat(6823634.024752475,9261), test['price_doc'].values)
train.price_doc.median()
rmsle(np.repeat(6000000,9261), test['price_doc'].values)
rmsle(np.repeat(6500000,9261), test['price_doc'].values)
#list(train.columns.values)
gb = train.groupby(['area_m'])



dfagg = pd.DataFrame()



# bayesian average

dfagg['avg_price_per_sqm'] = (5 * 6000000 + gb.price_doc.sum()) / (5 * 40 + gb.full_sq.sum())



dfagg['observations_count'] = gb.price_doc.count()

dfagg.reset_index(inplace=True)

dfagg.head()
test_merged = pd.merge(test, dfagg, how='left', on=['area_m'])

test_merged['avg_price_per_sqm'] = test_merged.avg_price_per_sqm.replace(np.NaN, 6823634.024752475)

test_merged['est_price'] = test_merged['avg_price_per_sqm'] * test_merged['full_sq']

test_merged.head()
rmsle(test_merged['est_price'].values, test_merged['price_doc'].values)
gb = train.groupby(['area_m', 'sub_area'])



dfagg = pd.DataFrame()

dfagg['avg_price_per_sqm'] = gb.price_doc.sum() / gb.full_sq.sum()

dfagg.reset_index(inplace=True)



test_merged = pd.merge(test, dfagg, how='left', on=['area_m', 'sub_area'])

test_merged['avg_price_per_sqm'] = test_merged.avg_price_per_sqm.replace(np.NaN, 6623634)

test_merged['est_price'] = test_merged['avg_price_per_sqm'] * test_merged['full_sq']



rmsle(test_merged['est_price'].values, test_merged['price_doc'].values)
train['dist'] = np.round(train['kremlin_km']/2)

test['dist'] = np.round(test['kremlin_km']/2)

gb = train.groupby(['dist'])



dfagg = pd.DataFrame()

dfagg['avg_price_per_sqm'] = (3 * 6000000.0 + gb.price_doc.sum()) / (3 * 60 + gb.full_sq.sum())

dfagg.reset_index(inplace=True)



test_merged = pd.merge(test, dfagg, how='left', on=['dist'])

test_merged['avg_price_per_sqm'] = test_merged.avg_price_per_sqm.replace(np.NaN, 6623634)

test_merged['est_price'] = test_merged['avg_price_per_sqm'] * test_merged['full_sq']



rmsle(test_merged['est_price'].values, test_merged['price_doc'].values)