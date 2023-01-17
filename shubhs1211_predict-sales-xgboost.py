# import relevant libraries

import numpy as np

import pandas as pd

import gc

import random

import xgboost as xgb

from sklearn.cross_validation import train_test_split
# Select what to skip (random sampling)

n = 125497040 # number of records in file

s = 1000 # sample size

select = sorted(random.sample(range(1, n + 1), s))

skip  = tuple(set(range(1, n + 1)) - set(select))
train = pd.read_csv("../input/train.csv",skiprows = skip)
train.head()
print(train.shape)
test  = pd.read_csv("../input/test.csv")

store = pd.read_csv("../input/stores.csv")

holiday = pd.read_csv("../input/holidays_events.csv")

item = pd.read_csv("../input/items.csv")

oil = pd.read_csv("../input/oil.csv")

transaction = pd.read_csv("../input/transactions.csv")
merged_train = pd.merge(train, store, on= "store_nbr")

merged_train = pd.merge(merged_train, item, on= "item_nbr")

merged_train = pd.merge(merged_train, holiday, on="date")

merged_train = pd.merge(merged_train, oil, on ="date")
merged_train.head()
train_items = pd.merge(train, item, how='inner')
train_items.head()
oil.head()
transaction.head()
item.head()
# no. of families (unique) 

print(len(item['family'].unique()))
holiday.head()
store.head()
# type of stores

print(len(store['type'].unique()))
test.head()
# Data Preprocessing

oil_nan = (oil.isnull().sum() / oil.shape[0]) * 100

oil_nan
store_nan = (store.isnull().sum() / store.shape[0]) * 100

store_nan
item_nan = (item.isnull().sum() / item.shape[0]) * 100

item_nan
train_nan = (train.isnull().sum() / train.shape[0]) * 100

train_nan
merged_train['onpromotion'] = merged_train['onpromotion'].fillna(2)

merged_train['onpromotion'] = merged_train['onpromotion'].replace(True,1)

merged_train['onpromotion'] = merged_train['onpromotion'].replace(False,0)
(merged_train['onpromotion'].unique())
merged_train['dcoilwtico'] = merged_train['dcoilwtico'].fillna(0)
merged_train['Year']  = merged_train['date'].apply(lambda x: int(str(x)[:4]))

merged_train['Month'] = merged_train['date'].apply(lambda x: int(str(x)[5:7]))

merged_train['date']  = merged_train['date'].apply(lambda x: (str(x)[8:]))





test['Year']  = test['date'].apply(lambda x: int(str(x)[:4]))

test['Month'] = test['date'].apply(lambda x: int(str(x)[5:7]))

test['date']  = test['date'].apply(lambda x: (str(x)[8:]))



train.head()
# create 2 copies of train_items

train_items1 = pd.merge(train, item, how='inner')

train_items2 = pd.merge(train, item, how='inner')
# train_items1

train_items1['date'] = pd.to_datetime(train_items1['date'], format = '%Y-%m-%d')

train_items1['day_item_purchased'] = train_items1['date'].dt.day

train_items1['month_item_purchased'] =train_items1['date'].dt.month

train_items1['quarter_item_purchased'] = train_items1['date'].dt.quarter

train_items1['year_item_purchased'] = train_items1['date'].dt.year

train_items1.drop('date', axis = 1, inplace = True)



# train_items2

train_items2['date'] = pd.to_datetime(train_items2['date'], format = '%Y-%m-%d')

train_items2['day_item_purchased'] = train_items2['date'].dt.day

train_items2['month_item_purchased'] = train_items2['date'].dt.month

train_items2['quarter_item_purchased'] = train_items2['date'].dt.quarter

train_items2['year_item_purchased'] = train_items2['date'].dt.year

train_items2.drop('date', axis = 1, inplace = True)
# train_items1

train_items1.loc[(train_items1.unit_sales<0),'unit_sales'] = 1 

train_items1['unit_sales'] =  train_items1['unit_sales'].apply(pd.np.log1p) 

train_items1['family'] = train_items1['family'].astype('category')

train_items1['onpromotion'] = train_items1['onpromotion'].astype('category')

train_items1['perishable'] = train_items1['perishable'].astype('category')

category_columns = train_items1.select_dtypes(['category']).columns

train_items1[category_columns] = train_items1[category_columns].apply(lambda x: x.cat.codes)



# train_items2

train_items2.loc[(train_items2.unit_sales<0),'unit_sales'] = 1 

train_items2['unit_sales'] =  train_items2['unit_sales'].apply(pd.np.log1p) 

train_items2['family'] = train_items2['family'].astype('category')

train_items2['onpromotion'] = train_items2['onpromotion'].astype('category')

train_items2['perishable'] = train_items2['perishable'].astype('category')

category_columns = train_items2.select_dtypes(['category']).columns

train_items2[category_columns] = train_items2[category_columns].apply(lambda x: x.cat.codes)
# train_items1

train_items1 = train_items1.drop(['unit_sales','family','class','perishable'], axis = 1)

train_items1.head()



# train_items2

train_items2 = train_items2.drop(['id','store_nbr','item_nbr','onpromotion', 'day_item_purchased','month_item_purchased','quarter_item_purchased','year_item_purchased','family','class','perishable'], axis = 1)

train_items2.head()
Xg_train, Xg_valid = train_test_split(train_items1, test_size=0.012, random_state=10)

Yg_train, Yg_valid = train_test_split(train_items2, test_size=0.012, random_state=10)

features1 = list(train_items1.columns.values)

features2 = list(train_items2.columns.values)
features1
features2
dtrain = xgb.DMatrix(Xg_train[features1], Yg_train[features2])

dvalid = xgb.DMatrix(Xg_valid[features1], Yg_valid[features2])
def rmspe(y, yhat):

    return np.sqrt(np.mean((yhat / y-1) ** 2))
def rmspe_xg(yhat, y):

    y = np.expm1(y.get_label())

    y1 = np.expm1(yhat)

    return "rmspe", rmspe(y, yhat)
params = {"objective": "reg:linear",

          "booster" : "gbtree",

          "eta": 0.3,

          "max_depth": 10,

          "subsample": 0.9,

          "colsample_bytree": 0.7,

          "silent": 1,

          "seed": 1301

          }

num_boost_round = 15

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
gbm = xgb.train(params, dtrain, num_boost_round, evals = watchlist,

  early_stopping_rounds = 5, feval = rmspe_xg, verbose_eval = True)
print("Validating")

yhat = gbm.predict(xgb.DMatrix(Xg_valid[features1]))

error = rmspe(Yg_valid.unit_sales.values, np.expm1(yhat))
test_copy = pd.read_csv("../input/test.csv")

test_copy.head()
test_copy['date'] = pd.to_datetime(test_copy['date'], format='%Y-%m-%d')

test_copy['day_item_purchased'] = test_copy['date'].dt.day

test_copy['month_item_purchased'] = test_copy['date'].dt.month

test_copy['quarter_item_purchased'] = test_copy['date'].dt.quarter

test_copy['year_item_purchased'] = test_copy['date'].dt.year

test_copy.drop('date', axis=1, inplace=True)
test_copy.head()
features1
train_items.head()
test_copy['onpromotion'] = test_copy['onpromotion'].astype('category')

cat_columns = test_copy.select_dtypes(['category']).columns

test_copy[cat_columns] = test_copy[cat_columns].apply(lambda x: x.cat.codes)
test_dmatrix = xgb.DMatrix(test_copy[features1])
test_prediction = gbm.predict(test_dmatrix)

print("Predictions")
result = pd.DataFrame({"id": test["id"], 'unit_sales': np.expm1(test_prediction)})

result.to_csv("submissionXG.csv", index=False)

print("Submission created")