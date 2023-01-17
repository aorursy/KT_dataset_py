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

train_df = train_df.sort_values(by='timestamp')



# split train / test

train = train_df[train_df['timestamp']< "2014-08-01"]

test  = train_df[train_df['timestamp']>= "2014-08-01"]
print(train.shape)

print(test.shape)
plt.scatter(train['id'], train['price_doc'], alpha=0.5, cmap='viridis')

plt.show()
train.head(1)
# moving average

# first let's average per day

#train['day'] = train['timestamp'].dt.day



#train['timestamp'].values

pd.to_datetime(train['timestamp'])



#train['moving_average'] = train['price_doc'].rolling(1000).mean()

#plt.scatter(train['id'], train['moving_average'])

#plt.show()
train.price_doc.mean()
rmsle(np.repeat(6823634.024752475,9261), test['price_doc'].values)
train.price_doc.median()
rmsle(np.repeat(6000000,9261), test['price_doc'].values)
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