import numpy as np

import pandas as pd

import sklearn

import xgboost as xgb



from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error

from sklearn.externals import joblib



import re

from itertools import product

import gc



import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

print(os.listdir("../input"))
# Check library versions

for p in [np, pd, sklearn, xgb]:

    print (p.__name__, p.__version__)
test_df = pd.read_csv('../input/test.csv', dtype={'ID': 'int32', 'shop_id': 'int32', 'item_id': 'int32'})

item_cat_df = pd.read_csv('../input/item_categories.csv', dtype={'item_category_name': 'str', 'item_category_id': 'int32'})

item_df = pd.read_csv('../input/items.csv', dtype={'item_name': 'str', 'item_id': 'int32', 'item_category_id': 'int32'})

shops_df= pd.read_csv('../input/shops.csv', dtype={'shop_name': 'str', 'shop_id': 'int32'})

train_df = pd.read_csv('../input/sales_train.csv', parse_dates=['date'], dtype={'date': 'str', 'date_block_num': 'int32', 'shop_id': 'int32', 'item_id': 'int32', 'item_price': 'float32', 'item_cnt_day': 'int32'})
print(f'''Shapes:

Item categories: {item_cat_df.shape}

Items: {item_df.shape}

Shops: {shops_df.shape}

Train set: {train_df.shape}

Test set: {test_df.shape}''')
#Sale date minimum 

#Sale date maximum

print(train_df['date'].min())

print(train_df['date'].max())
print("Shape of train data : {}, Shape of test data : {}".format(train_df.shape, test_df.shape))
train = train_df.join(item_df, on='item_id', rsuffix='_').join(shops_df, on='shop_id', rsuffix='_').join(item_cat_df, on='item_category_id', rsuffix='_').drop(['item_id_', 'shop_id_', 'item_category_id_'], axis=1)

train.head()
train.describe()
#Data Cleaning

train = train.query('item_price > 0')
#Data leakages

#About data leakages I'll only be using only the "shop_id" and "item_id" that appear on the test set.



test_shop_ids = test_df['shop_id'].unique()

test_item_ids = test_df['item_id'].unique()

# Only shops that exist in test set.

lk_train = train[train['shop_id'].isin(test_shop_ids)]

# Only items that exist in test set.

lk_train = lk_train[lk_train['item_id'].isin(test_item_ids)]



print('Data set size before leaking:', train.shape[0])

print('Data set size after leaking:', lk_train.shape[0])
# Select only useful features.

train_monthly = lk_train[['date', 'date_block_num', 'shop_id', 'item_category_id', 'item_id', 'item_price', 'item_cnt_day']]

# Group by month in this case "date_block_num" and aggregate features.

train_monthly = train_monthly.sort_values('date').groupby(['date_block_num', 'shop_id', 'item_category_id', 'item_id'], as_index=False)

train_monthly = train_monthly.agg({'item_price':['sum', 'mean'], 'item_cnt_day':['sum', 'mean','count']})

# Rename features.

train_monthly.columns = ['date_block_num', 'shop_id', 'item_category_id', 'item_id', 'item_price', 'mean_item_price', 'item_cnt', 'mean_item_cnt', 'transactions']
# Build a data set with all the possible combinations of ['date_block_num','shop_id','item_id'] so we won't have missing records.

shop_ids = train_monthly['shop_id'].unique()

item_ids = train_monthly['item_id'].unique()

empty_df = []

for i in range(34):

    for shop in shop_ids:

        for item in item_ids:

            empty_df.append([i, shop, item])

    

empty_df = pd.DataFrame(empty_df, columns=['date_block_num','shop_id','item_id'])
# Merge the train set with the complete set (missing records will be filled with 0).

train_monthly = pd.merge(empty_df, train_monthly, on=['date_block_num','shop_id','item_id'], how='left')

train_monthly.fillna(0, inplace=True)
train_monthly.head().T
# Extract time based features.

train_monthly['year'] = train_monthly['date_block_num'].apply(lambda x: ((x//12) + 2013))

train_monthly['month'] = train_monthly['date_block_num'].apply(lambda x: (x % 12))
#Remove outliers

train_monthly = train_monthly.query('item_cnt >= 0 and item_cnt <= 20 and item_price < 400000')

train_monthly['item_cnt_month'] = train_monthly.sort_values('date_block_num').groupby(['shop_id', 'item_id'])['item_cnt'].shift(-1)
train_set = train_monthly.query('date_block_num >= 3 and date_block_num < 28').copy()

validation_set = train_monthly.query('date_block_num >= 28 and date_block_num < 33').copy()

test_set = train_monthly.query('date_block_num == 33').copy()





train_set.dropna(inplace=True)

validation_set.dropna(inplace=True)



print('Train set records:', train_set.shape[0])

print('Validation set records:', validation_set.shape[0])

print('Test set records:', test_set.shape[0])



print('Train set records: %s (%.f%% of complete data)' % (train_set.shape[0], ((train_set.shape[0]/train_monthly.shape[0])*100)))

print('Validation set records: %s (%.f%% of complete data)' % (validation_set.shape[0], ((validation_set.shape[0]/train_monthly.shape[0])*100)))
# Create train and validation sets and labels. 

X_train = train_set.drop(['item_cnt_month', 'date_block_num'], axis=1)

Y_train = train_set['item_cnt_month'].astype(int)

X_validation = validation_set.drop(['item_cnt_month', 'date_block_num'], axis=1)

Y_validation = validation_set['item_cnt_month'].astype(int)
# Integer features (used by catboost model).

int_features = ['shop_id', 'item_id', 'year', 'month']



X_train[int_features] = X_train[int_features].astype('int32')

X_validation[int_features] = X_validation[int_features].astype('int32')
latest_records = pd.concat([train_set, validation_set]).drop_duplicates(subset=['shop_id', 'item_id'], keep='last')

X_test = pd.merge(test_df, latest_records, on=['shop_id', 'item_id'], how='left', suffixes=['', '_'])

X_test['year'] = 2015

X_test['month'] = 9

X_test.drop('item_cnt_month', axis=1, inplace=True)

X_test[int_features] = X_test[int_features].astype('int32')

X_test = X_test[X_train.columns]
#replace missing val

sets = [X_train, X_validation, X_test]



# This was taking too long.

# Replace missing values with the median of each item.

# for dataset in sets:

#     for item_id in dataset['item_id'].unique():

#         for column in dataset.columns:

#             item_median = dataset[(dataset['item_id'] == item_id)][column].median()

#             dataset.loc[(dataset[column].isnull()) & (dataset['item_id'] == item_id), column] = item_median



# Replace missing values with the median of each shop.            

for dataset in sets:

    for shop_id in dataset['shop_id'].unique():

        for column in dataset.columns:

            shop_median = dataset[(dataset['shop_id'] == shop_id)][column].median()

            dataset.loc[(dataset[column].isnull()) & (dataset['shop_id'] == shop_id), column] = shop_median

            

# Fill remaining missing values on test set with mean.

X_test.fillna(X_test.mean(), inplace=True)
# I'm dropping "item_category_id", we don't have it on test set and would be a little hard to create categories for items that exist only on test set.

X_train.drop(['item_category_id'], axis=1, inplace=True)

X_validation.drop(['item_category_id'], axis=1, inplace=True)

X_test.drop(['item_category_id'], axis=1, inplace=True)

X_train.head()
from xgboost import XGBRegressor

from xgboost import plot_importance

# Use only part of features on XGBoost.

xgb_features = ['item_cnt','item_price', 

                'mean_item_price', 'mean_item_cnt']



xgb_train = X_train[xgb_features]

xgb_val = X_validation[xgb_features]

xgb_test = X_test[xgb_features]

xgb_model = XGBRegressor(max_depth=8, 

                         n_estimators=500, 

                         min_child_weight=1000,  

                         colsample_bytree=0.7, 

                         subsample=0.7, 

                         eta=0.3, 

                         seed=0)

xgb_model.fit(xgb_train, 

              Y_train, 

              eval_metric="rmse", 

              eval_set=[(xgb_train, Y_train), (xgb_val, Y_validation)], 

              verbose=20, 

              early_stopping_rounds=20)
plt.rcParams["figure.figsize"] = (15, 6)

plot_importance(xgb_model)

plt.show()
xgb_train_pred = xgb_model.predict(xgb_train)

xgb_val_pred = xgb_model.predict(xgb_val)

xgb_test_pred = xgb_model.predict(xgb_test)
print('Train rmse:', np.sqrt(mean_squared_error(Y_train, xgb_train_pred)))

print('Validation rmse:', np.sqrt(mean_squared_error(Y_validation, xgb_val_pred)))
prediction_df = pd.DataFrame(test_df['ID'], columns=['ID'])

prediction_df['item_cnt_month'] = xgb_test_pred.clip(0., 20.)

prediction_df.to_csv('submission.csv', index=False)

prediction_df.head(10)
#Check for null data

train_df = train_df.sort_values('date')

train_df.isnull().sum()
#Store turnover by date agregation calcul

train_TS= train_df.groupby('date')['item_price'].sum().reset_index()



train_TS = train_TS.set_index('date')

train_TS.index
train_TS.plot(figsize=(25, 6))

plt.show()