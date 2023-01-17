# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import sklearn
sklearn.__version__
# Linear regression on a categorical variable using one-hot and dummy codes
from sklearn import linear_model

# Define a toy dataset of apartment rental prices in 
# New York, San Francisco, and Seattle
df = pd.DataFrame({
     'City': ['SF', 'SF', 'SF', 'NYC', 'NYC', 'NYC', 
              'Seattle', 'Seattle', 'Seattle'],
     'Rent': [3999, 4000, 4001, 3499, 3500, 3501, 2499, 2500, 2501]
})
df['Rent'].mean()
# Convert the categorical variables in the DataFrame to one-hot encoding
# and fit a linear regression model
one_hot_df = pd.get_dummies(df, prefix=['city'])
one_hot_df
model = linear_model.LinearRegression()
model.fit(one_hot_df[['city_NYC', 'city_SF', 'city_Seattle']],
           one_hot_df['Rent'])
model.coef_
model.intercept_
dummy_df = pd.get_dummies(df, prefix=['city'], drop_first=True)
dummy_df
model.fit(dummy_df[['city_SF', 'city_Seattle']], dummy_df['Rent'])
model.coef_

model.intercept_
# bias coefficient
df[df['City'] == 'NYC'].mean()
# The coefficient for the x1 feature
df[df['City'] == 'NYC'].mean() - 3500
# The coefficient for the x2 feature
df[df['City'] == 'SF'].mean() - 3500
# The coefficient for the x3 feature
df[df['City'] == 'Seattle'].mean() - 3500
effect_df = dummy_df.copy()
effect_df.loc[3:5, ['city_SF', 'city_Seattle']] = -1.0
effect_df
model.fit(effect_df[['city_SF', 'city_Seattle']], effect_df['Rent'])
model.coef_
model.intercept_
import json
# Load Yelp reviews data
with open('../input/yelp-dataset/yelp_academic_dataset_review.json') as review_file:
    review_df = pd.DataFrame([json.loads(next(review_file)) for x in range(10000)])
# Define m as equal to the unique number of business_ids
m = len(review_df.business_id.unique())
m
from sklearn.feature_extraction import FeatureHasher

h = FeatureHasher(n_features=m, input_type='string')
f = h.transform(review_df['business_id'])
# How does this affect feature interpretability?
review_df['business_id'].unique().tolist()[0:5]
f.toarray()
# Not great. BUT, let's see the storage size of our features.
from sys import getsizeof
print('Our pandas Series, in bytes: ', getsizeof(review_df['business_id']))
print('Our hashed numpy array, in bytes: ', getsizeof(f))
# train_subset data is first 10K rows of 6+GB set
df = pd.read_csv('../input/avazu-ctr-prediction/train/train.csv', nrows=10000)
df.head()
# How many unique features should we have after?
len(df['device_id'].unique())
def click_counting(x, bin_column):
    clicks = pd.Series(x[x['click'] > 0][bin_column].value_counts(), name='clicks')
    no_clicks = pd.Series(x[x['click'] < 1][bin_column].value_counts(), name='no_clicks')
    
    counts = pd.DataFrame([clicks,no_clicks]).T.fillna('0')
    counts['total_clicks'] = counts['clicks'].astype('int64') + counts['no_clicks'].astype('int64')
    return counts

def bin_counting(counts):
    counts['N+'] = counts['clicks']\
                    .astype('int64')\
                    .divide(counts['total_clicks'].astype('int64'))
    counts['N-'] = counts['no_clicks']\
                    .astype('int64')\
                    .divide(counts['total_clicks'].astype('int64'))
    counts['log_N+'] = counts['N+'].divide(counts['N-'])
    # If we wanted to only return bin-counting properties, 
    # we would filter here
    bin_counts = counts.filter(items= ['N+', 'N-', 'log_N+'])
    return counts, bin_counts
bin_column = 'device_id'
device_clicks = click_counting(df.filter(items=[bin_column, 'click']), bin_column)
device_all, device_bin_counts = bin_counting(device_clicks.copy())

device_clicks.head()
device_all.head()
device_bin_counts.head()
len(device_bin_counts)
device_all.sort_values(by = 'total_clicks', ascending=False).head(4)
