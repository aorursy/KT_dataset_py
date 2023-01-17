from catboost import CatBoostRegressor

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

import itertools

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import mean_squared_error as mse

from sklearn.preprocessing import LabelEncoder
path = '../input/competitive-data-science-predict-future-sales/'
train = pd.read_csv(path+'sales_train.csv')

test = pd.read_csv(path+'test.csv')

items = pd.read_csv(path+'items.csv')

categ = pd.read_csv(path+'item_categories.csv')

shops = pd.read_csv(path+'shops.csv')
test.insert(1, 'date_block_num', 34)

test.head()
train.drop('item_price', axis = 1, inplace = True)

train.head()
shops.drop(index = [10,57,58], inplace = True)

shops.update(pd.Series(['СергиевПосад ТЦ "7Я"'], name='shop_name', index=[46]))

shops['city'] = shops['shop_name'].apply(lambda x : x.split(' ')[0] if x.split(' ')[0] != '!Якутск' else 'Якутск')

shops['qty'] = shops['city'].apply(lambda x: len(shops[shops['city']==x]))

shops.drop('shop_name', axis = 1, inplace=True)

le = LabelEncoder()

shops['city'] = le.fit_transform(shops['city'])
f57 = train[(train['shop_id'] == 57)].index

f58 = train[(train['shop_id'] == 58)].index

f10 = train[(train['shop_id'] == 10)].index



i57 = test[(test['shop_id'] == 57)].index

i58 = test[(test['shop_id'] == 58)].index

i10 = test[(test['shop_id'] == 10)].index



train.update(pd.Series(0, name='shop_id', index=f57))

train.update(pd.Series(1, name='shop_id', index=f58))

train.update(pd.Series(11, name='shop_id', index=f10))



test.update(pd.Series(0, name='shop_id', index=i57))

test.update(pd.Series(1, name='shop_id', index=i58))

test.update(pd.Series(11, name='shop_id', index=i10))
categ['gen_categ'] = categ['item_category_name'].apply(lambda x: x.split(' ')[0])

categ['gen_categ'] = le.fit_transform(categ['gen_categ'])

categ.drop('item_category_name', axis = 1, inplace = True)
items.drop('item_name', axis = 1,inplace = True)
temp = train.groupby(['date_block_num','shop_id','item_id'], as_index=False).sum()
temp['shop_id'] = np.int8(temp['shop_id'])

temp['item_cnt_day'] = np.int16(temp['item_cnt_day'])
test.head()
it_cat = items.merge(categ, on = 'item_category_id')
test_shop = temp.merge(shops, on = 'shop_id')
data = test_shop.merge(it_cat, on = 'item_id')
data.head()
test = test.drop('ID', axis = 1)

test = test.merge(shops, on = 'shop_id').merge(it_cat, on = 'item_id')

test['shop_id'] = np.int8(test['shop_id'])

test.head()
cat = CatBoostRegressor(iterations=800, learning_rate=0.05, loss_function="RMSE")
X = data.drop('item_cnt_day', axis = 1).values

y = data['item_cnt_day'].values
skf = StratifiedKFold(n_splits = 3)
X_train, X_test, y_train, y_test = train_test_split(X,y)
for train_index, test_index in skf.split(X, y):

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    cat.fit(X_train, y_train, eval_set = (X_test, y_test),plot = True, early_stopping_rounds=5, verbose = False)
cat.fit(X,y,plot = True, early_stopping_rounds=5, verbose = False)
result = cat.predict(test)

result
#pd.DataFrame({'ID': test.index,'item_cnt_month':result}).to_csv('../output/kaggle/working/sub.csv',index = False)