# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')

shops = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')
cat = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')

train = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv',)

test  = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
cat.head()
cat_list = list(cat.item_category_name)



for i in range(1,8):

    cat_list[i] = 'Access'



for i in range(10,18):

    cat_list[i] = 'Consoles'



for i in range(18,25):

    cat_list[i] = 'Consoles Games'



for i in range(26,28):

    cat_list[i] = 'phone games'



for i in range(28,32):

    cat_list[i] = 'CD games'



for i in range(32,37):

    cat_list[i] = 'Card'



for i in range(37,43):

    cat_list[i] = 'Movie'



for i in range(43,55):

    cat_list[i] = 'Books'



for i in range(55,61):

    cat_list[i] = 'Music'



for i in range(61,73):

    cat_list[i] = 'Gifts'



for i in range(73,79):

    cat_list[i] = 'Soft'



cat['cats'] = cat_list

cat.head()
train.head()
train['date'] = pd.to_datetime(train.date,format="%d.%m.%Y")

train.head()
pivot = train.pivot_table(index=['shop_id','item_id'], 

                            columns='date_block_num', 

                            values='item_cnt_day',

                            aggfunc='sum').fillna(0.0)

pivot.head()
train_cleaned = pivot.reset_index()

train_cleaned['shop_id']= train_cleaned.shop_id.astype('str')

train_cleaned['item_id']= train_cleaned.item_id.astype('str')

item_to_cat = items.merge(cat[['item_category_id','cats']], 

                                how="inner", 

                                on="item_category_id")[['item_id','cats']]

item_to_cat[['item_id']] = item_to_cat.item_id.astype('str')

train_cleaned = train_cleaned.merge(item_to_cat, how="inner", on="item_id")

train_cleaned.head()
from sklearn import preprocessing



le = preprocessing.LabelEncoder()

train_cleaned[['cats']] = le.fit_transform(train_cleaned.cats)

train_cleaned = train_cleaned[['shop_id', 'item_id', 'cats'] + list(range(34))]

train_cleaned.head()
X_train = train_cleaned.iloc[:, :-1].values

print(X_train.shape)

X_train[:3]
y_train = train_cleaned.iloc[:, -1].values

print(y_train.shape)
import xgboost as xgb

param = {'max_depth':13, 

         'subsample':1,

         'min_child_weight':0.5,  

         'eta':1,

         'num_round':1000, 

         'seed':42, 

         'silent':0,

         'eval_metric':'rmse',

         'early_stopping_rounds':100

        }



progress = dict()

xgbtrain = xgb.DMatrix(X_train, y_train)

watchlist  = [(xgbtrain,'train-rmse')]

bst = xgb.train(param, xgbtrain)
from sklearn.metrics import mean_squared_error 

preds = bst.predict(xgb.DMatrix(X_train))



rmse = np.sqrt(mean_squared_error(preds, y_train))

print(rmse)
xgb.plot_importance(bst);
test.head()
test['shop_id']= test.shop_id.astype('str')

test['item_id']= test.item_id.astype('str')



test = test.merge(train_cleaned, how = "left", on = ["shop_id", "item_id"]).fillna(0.0)

test.head()
d = dict(zip(test.columns[4:], list(np.array(list(test.columns[4:])) - 1)))



test  = test.rename(d, axis = 1)

test.head()
X_test = test.drop(['ID', -1], axis=1).values

print(X_test.shape)

preds = bst.predict(xgb.DMatrix(X_test))

print(preds.shape)
sub = pd.DataFrame({'ID':test.ID, 'item_cnt_month': preds.clip(0. ,20.)})

sub.to_csv('submission.csv',index=False)
y_all = np.append(y_train, preds)

y_all.shape
X_all = np.concatenate((X_train, X_test), axis=0)

X_all.shape
param = {'max_depth':13,

         'subsample':1, 

         'min_child_weight':0.5, 

         'eta':1,

         'num_round':1000, 

         'seed':42,

         'silent':0,

         'eval_metric':'rmse',

         'early_stopping_rounds':100

        }



progress = dict()

xgbtrain = xgb.DMatrix(X_all, y_all)

watchlist  = [(xgbtrain,'train-rmse')]

bst = xgb.train(param, xgbtrain)
from sklearn.metrics import mean_squared_error 

preds = bst.predict(xgb.DMatrix(X_train))



rmse = np.sqrt(mean_squared_error(preds, y_train))

print(rmse)
preds = bst.predict(xgb.DMatrix(X_test))

print(preds.shape)

sub = pd.DataFrame({'ID':test.ID, 'item_cnt_month': preds.clip(0. ,20.)})

sub.to_csv('submission2.csv',index=False)