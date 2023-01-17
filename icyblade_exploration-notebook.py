import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from PIL import Image

%matplotlib inline
ROOT = '../input/'  # ROOT of our data. You can change this on your environment. 

train_raw = pd.read_json(ROOT+'train.json')

test_raw = pd.read_json(ROOT+'test.json')
train_raw.sample(5).T
Image.open(ROOT+'/images_sample/6811957/6811957_33d08c8dc440c89bccc8d9889c5485a6.jpg')
print('Total: %s' % train_raw.shape[0])

print(train_raw[['building_id', 'listing_id', 'manager_id']].T.apply(lambda row: row.nunique(), axis=1))
train_raw.groupby('interest_level').listing_id.count().loc[['low', 'medium', 'high']].plot(kind='bar')

plt.xticks(rotation='horizontal')

plt.show()
fig, axes = plt.subplots(1,2)

train_raw.groupby('bathrooms').listing_id.count().plot(kind='bar', ax=axes[0])

train_raw.groupby('bedrooms').listing_id.count().plot(kind='bar', ax=axes[1])

plt.show()
features = pd.DataFrame({'feature': [j for i in train_raw.features.values for j in i]})

features['dummy'] = 1

features.groupby('feature').count().sort_values('dummy', ascending=False)
print(train_raw.price.describe(percentiles=[0.01, 0.99]))
price_lower_bound = 1475.00

price_higher_bound = 13000.00

train_raw.query('%s<=price and price<=%s' % (price_lower_bound, price_higher_bound)).price.hist(bins=50)

plt.show()
# remove skewness

from scipy.stats import boxcox

from sklearn.preprocessing import MinMaxScaler

data = train_raw.query('%s<=price and price<=%s' % (price_lower_bound, price_higher_bound)).price.values.reshape(-1, 1)

price_scaler = MinMaxScaler(feature_range=(1, 2)).fit(data.astype(float))

d, price_bc_coeff = boxcox(price_scaler.transform(data))

price_bc_coeff = price_bc_coeff[0]

print('BoxCox lambda: %s' % price_bc_coeff)

plt.hist(d, bins=50)

plt.show()