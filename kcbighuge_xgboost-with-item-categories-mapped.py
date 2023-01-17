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
# Get the data
items_df = pd.read_csv('../input/items.csv')
shops_df = pd.read_csv('../input/shops.csv')

cat_df = pd.read_csv('../input/item_categories.csv')
train_df = pd.read_csv('../input/sales_train.csv',)
test_df  = pd.read_csv('../input/test.csv')
cat_df.head()
# Get english category names
cat_list = list(cat_df.item_category_name)

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

cat_df['cats'] = cat_list
cat_df.head()
train_df.head()
# Adjust the date format
train_df['date'] = pd.to_datetime(train_df.date,format="%d.%m.%Y")
train_df.head()
# Pivot the data by month
pivot_df = train_df.pivot_table(index=['shop_id','item_id'], 
                            columns='date_block_num', 
                            values='item_cnt_day',
                            aggfunc='sum').fillna(0.0)
pivot_df.head()
# Convert id values to string
train_cleaned_df = pivot_df.reset_index()
train_cleaned_df['shop_id']= train_cleaned_df.shop_id.astype('str')
train_cleaned_df['item_id']= train_cleaned_df.item_id.astype('str')

# Join with categories
item_to_cat_df = items_df.merge(cat_df[['item_category_id','cats']], 
                                how="inner", 
                                on="item_category_id")[['item_id','cats']]
item_to_cat_df[['item_id']] = item_to_cat_df.item_id.astype('str')

train_cleaned_df = train_cleaned_df.merge(item_to_cat_df, how="inner", on="item_id")
train_cleaned_df.head()
# Encode Categories
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
train_cleaned_df[['cats']] = le.fit_transform(train_cleaned_df.cats)
train_cleaned_df = train_cleaned_df[['shop_id', 'item_id', 'cats'] + list(range(34))]
train_cleaned_df.head()
# Create input features, leaving out last column
X_train = train_cleaned_df.iloc[:, :-1].values
print(X_train.shape)
X_train[:3]
# Create target to predict
y_train = train_cleaned_df.iloc[:, -1].values
print(y_train.shape)
# Set parameters and train the model
# See: http://xgboost.readthedocs.io/en/latest/parameter.html
import xgboost as xgb
param = {'max_depth':12,  # originally 10
         'subsample':1,  # 1
         'min_child_weight':0.5,  # 0.5
         'eta':0.3,
         'num_round':1000, 
         'seed':0,  # 1
         'silent':0,
         'eval_metric':'rmse',
         'early_stopping_rounds':100
        }

progress = dict()
xgbtrain = xgb.DMatrix(X_train, y_train)
watchlist  = [(xgbtrain,'train-rmse')]
bst = xgb.train(param, xgbtrain)
# Get predictions and rmse score
from sklearn.metrics import mean_squared_error 
preds = bst.predict(xgb.DMatrix(X_train))

rmse = np.sqrt(mean_squared_error(preds, y_train))
print(rmse)
# Look at the most important features
xgb.plot_importance(bst);
test_df.head()
# Convert values to string
test_df['shop_id']= test_df.shop_id.astype('str')
test_df['item_id']= test_df.item_id.astype('str')

# Merge the monthly item count data
test_df = test_df.merge(train_cleaned_df, how = "left", on = ["shop_id", "item_id"]).fillna(0.0)
test_df.head()
# Adjust the labels for item count columns
d = dict(zip(test_df.columns[4:], list(np.array(list(test_df.columns[4:])) - 1)))

test_df  = test_df.rename(d, axis = 1)
test_df.head()
# Drop columns from test data and get predictions
X_test = test_df.drop(['ID', -1], axis=1).values
print(X_test.shape)

preds = bst.predict(xgb.DMatrix(X_test))
print(preds.shape)
# Clip to values 0 - 20 and create submission file
sub_df = pd.DataFrame({'ID':test_df.ID, 'item_cnt_month': preds.clip(0. ,20.)})
sub_df.to_csv('submission.csv',index=False)
# Combine test set predictions with training set labels
y_all = np.append(y_train, preds)
y_all.shape
# Combine training set and test set
X_all = np.concatenate((X_train, X_test), axis=0)
X_all.shape
# Set parameters and train the model
# See: http://xgboost.readthedocs.io/en/latest/parameter.html
import xgboost as xgb
param = {'max_depth':12,  # originally 10
         'subsample':1,  # 1
         'min_child_weight':0.5,  # 0.5
         'eta':0.3,
         'num_round':1000, 
         'seed':42,  # 1
         'silent':0,
         'eval_metric':'rmse',
         'early_stopping_rounds':100
        }

progress = dict()
xgbtrain = xgb.DMatrix(X_all, y_all)
watchlist  = [(xgbtrain,'train-rmse')]
bst = xgb.train(param, xgbtrain)
# Get predictions and rmse score
from sklearn.metrics import mean_squared_error 
preds = bst.predict(xgb.DMatrix(X_train))

rmse = np.sqrt(mean_squared_error(preds, y_train))
print(rmse)
# Get test predictions
preds = bst.predict(xgb.DMatrix(X_test))
print(preds.shape)

# Clip to values 0 - 20 and create second submission file 
sub_df = pd.DataFrame({'ID':test_df.ID, 'item_cnt_month': preds.clip(0. ,20.)})
sub_df.to_csv('submission2.csv',index=False)