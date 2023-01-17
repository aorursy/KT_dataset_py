import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



import gc

import numba
item_cat = pd.read_csv('../input/item_categories.csv')

items = pd.read_csv('../input/items.csv')

shops = pd.read_csv('../input/shops.csv')

train = pd.read_csv('../input/sales_train.csv')

test = pd.read_csv('../input/test.csv')

sample = pd.read_csv('../input/sample_submission.csv')
print('Shape of {}: {}'.format('items', items.shape))

print('Shape of {}: {}'.format('item_cat', item_cat.shape))

print('Shape of {}: {}'.format('shops', shops.shape))

print('Shape of {}: {}'.format('train', train.shape))

print('Shape of {}: {}'.format('test', test.shape))

print('Shape of {}: {}'.format('sample', sample.shape))
print("Number of unique entities in each:")

print('{}------:\n{}'.format('items', items.nunique()))

print('{}------:\n{}'.format('item_cat', item_cat.nunique()))

print('{}------:\n{}'.format('shops', shops.nunique()))

print('{}------:\n{}'.format('train', train.nunique()))

print('{}------:\n{}'.format('test', test.nunique()))

print('{}------:\n{}'.format('sample', sample.nunique()))
train.head()
test.head()
sample.head()
item_cat.head()
items.head()
shops.head()
print("Any null in train? \n{}".format(train.isna().sum()))

print("Any null in test? \n{}".format(test.isna().sum()))
train['item_cat_id'] = train['item_id'].map(get_item_cat)

train_cat_grouped = train[['item_cat_id', 'item_price', 'item_cnt_day']].groupby(by=['item_cat_id'], as_index=False).sum()

train_store_grouped = train[['shop_id', 'item_cnt_day']].groupby(by=['shop_id'], as_index=False).sum()



train['date'] = pd.to_datetime(train['date'])

train['month'] = train['date'].map(lambda x: x.month)

train_month_grouped = df[['month', 'item_cnt_day']].groupby(by=['month'], as_index=False).sum()

# Lets visualize the trends between all the features of interest.

fig = plt.figure(figsize=(30,18))

ax1 = plt.subplot(4,1,1)

ax1.bar(x=train_store_grouped['shop_id'], height=train_store_grouped['item_cnt_day'], width=0.8)

ax1.set_xticklabels(labels=train_store_grouped['shop_id'], rotation=90)

ax1.set_xlabel("Shop ID")

ax1.set_ylabel("Total Items Sold")

ax1.set_title("Shop vs Total Items Sold")



ax2 = plt.subplot(4,1,2)

ax2.bar(x=train_cat_grouped['item_cat_id'], height=train_cat_grouped['item_cnt_day'], width=0.8)

ax2.set_xticklabels(labels=train_cat_grouped['item_cat_id'], rotation=90)

ax2.set_xlabel("Cat ID")

ax2.set_ylabel("Total Items Sold")

ax2.set_title("Cat vs Total Items Sold")



ax3 = plt.subplot(4,1,3)

ax3.bar(x=train_cat_grouped['item_cat_id'], height=train_cat_grouped['item_price'], width=0.8)

ax3.set_xticklabels(labels=train_cat_grouped['item_cat_id'], rotation=90)

ax3.set_xlabel("Cat ID")

ax3.set_ylabel("Total Money Made")

ax3.set_title("Cat vs Total Money Made")



ax3 = plt.subplot(4,1,4)

ax3.bar(x=train_month_grouped['month'], height=train_month_grouped['item_cnt_day'], width=0.8)

ax3.set_xticklabels(labels=train_month_grouped['month'], rotation=90)

ax3.set_xlabel("Month")

ax3.set_ylabel("Total item sold")

ax3.set_title("Month vs Total Money Made")

fig.tight_layout()

fig.show()
train_store_cat_grouped = df[['shop_id','item_cat_id', 'item_cnt_day']].groupby(by=['shop_id','item_cat_id'], as_index=False).sum()

train_store_cat_grouped["shop_item_id"] = train_store_cat_grouped["shop_id"].map(str) + " / " +train_store_cat_grouped["item_cat_id"].map(str)



fig = plt.figure(figsize=(30,10))

ax = plt.subplot(1,1,1)

ax.bar(x=train_store_cat_grouped["shop_item_id"], height=train_store_cat_grouped['item_cnt_day'], width=0.5)

ax.set_xticklabels(labels=train_store_cat_grouped["shop_item_id"], rotation=90)

ax.set_xlabel("Shop Cat ID")

ax.set_ylabel("Total Items Sold")

ax.set_title("Shop Cat vs Total Items Sold")

fig.show()
# Lets get the average price of all the items

avg_price_of_item = train[['item_id','item_price']].groupby(by=['item_id'], as_index=False).mean()
# Thanks to Denis Larlonov for discovering that each shop name comes with the city name in it. We can extract this and make another feature called city.

shops['city'] = shops['shop_name'].str.split(' ').str.get(0)



# Lets convert the city names to integer values

shops["city"] = shops["city"].astype('category')

shops["city_id"] = shops["city"].cat.codes

shops.drop(columns=['shop_name', 'city'], inplace=True)
# Lets group all the items by month now

train = train.groupby(['date_block_num', 'shop_id', 'item_id'], as_index=False).sum()

# Lets make a new feature to represent the months

train['month'] = (train['date_block_num'] + 1) % 12

# Finally we will combine item category to the train dataset by using 'item_id'

train = pd.merge(left=train, right=items, how='left', on='item_id')

train['item_category_id'].fillna(-1, inplace=True)

# Lets add the city id to the training set

train = pd.merge(left=train, right=shops, how='left', on='shop_id')

train['city_id'].fillna(-1, inplace=True)

# Lets create another train data frame but we are going to add 1 to date num block to create a lagged result

train_lagged = train[['date_block_num', 'shop_id', 'item_id', 'item_cnt_day']].copy(deep=True)

train_lagged['date_block_num'] = train_lagged['date_block_num'] + 1

train_lagged.rename(columns={'item_cnt_day' : 'lagged_item_cnt'}, inplace=True)

train = pd.merge(left=train, right=train_lagged, how='left', on=['date_block_num', 'shop_id', 'item_id'])



# We will combine item category to the test dataset by using 'item_id'

test = pd.merge(left=test, right=items, how='inner', on='item_id')

# Fill any missing category with value with -1

test['item_category_id'].fillna(-1, inplace=True)

# Add the average price from the train set to test set

test = pd.merge(left=test, right=avg_price_of_item, how='left', on='item_id')

# Fill the missing item price with the global average value

test['item_price'].fillna(test['item_price'].mean(), inplace=True)

# We know that the test set is november

test['month'] = 11

test['date_block_num'] = 34

# Lets add the city id to the test set

test = pd.merge(left=test, right=shops, how='inner', on='shop_id')

test['city_id'].fillna(-1, inplace=True)

# Lets create another train data frame but we are going to add 1 to date num block to create a lagged result

test = pd.merge(left=test, right=train_lagged, how='left', on=['date_block_num', 'shop_id', 'item_id'])



train.fillna(-1, inplace=True)

test.fillna(-1, inplace=True)



# Drop the item names since item_id already stores the uniquness

train.drop(columns=['item_name'], inplace=True)

test.drop(columns=['item_name'], inplace=True)



for col in train.columns:

    train[col] = pd.to_numeric(train[col], downcast='integer')



for col in test.columns:

    test[col] = pd.to_numeric(test[col], downcast='integer')

"""

from sklearn.feature_extraction import FeatureHasher

cols = {

    0:{

        'name': 'shop_id',

        'n_features': 15

    },

    1:{

        'name': 'item_id',

        'n_features': 10

    },

    2:{

        'name': 'item_cat_id',

        'n_features': 21

    },

    3:{

        'name': 'month',

        'n_features': 6

    }

}

def create_hashed_features(df):

    for i in range(len(cols)):

        fh = FeatureHasher(n_features=cols[i]['n_features'], input_type='string')

        hashed_features = fh.fit_transform(df[cols[i]['name']].map(str))

        hashed_features = hashed_features.toarray()

        df = pd.concat([df.drop(columns=[cols[i]['name']]), pd.DataFrame(hashed_features)], axis=1)

    return df

  

train_hashed = create_hashed_features(train)

#test_hashed = create_hashed_features(test)

"""
"""

from sklearn.feature_selection import RFE



validation = train.loc[train['date_block_num'] > 32]

train = train.loc[train['date_block_num'] <= 32]



all_y = train['item_cnt_day']

all_X = train.drop(columns=['item_cnt_day'])



rf = RandomForestClassifier()

#selector = RFE(rf)

#selector.fit(all_X,all_y)

"""
import xgboost as xgb



from sklearn.metrics import mean_squared_error

from math import sqrt
all_X = train.drop(columns=['item_cnt_day'])

all_y = train['item_cnt_day']
model = xgb.XGBRegressor()

model.fit(all_X, all_y)
test_df = test[all_X.columns]

predictions = model.predict(test_df)
submission = pd.DataFrame()

submission['ID'] = test['ID']

submission['item_cnt_month'] = predictions

submission['item_cnt_month'] = submission['item_cnt_month'].round(0)
submission.to_csv('submission.csv', index=False)