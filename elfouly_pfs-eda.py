import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
item_categories = pd.read_csv("../input/competitive-data-science-predict-future-sales/item_categories.csv")

items = pd.read_csv("../input/competitive-data-science-predict-future-sales/items.csv")

sales_train = pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")

sample_submission = pd.read_csv("../input/competitive-data-science-predict-future-sales/sample_submission.csv")

shops = pd.read_csv("../input/competitive-data-science-predict-future-sales/shops.csv")

test = pd.read_csv("../input/competitive-data-science-predict-future-sales/test.csv")
print(f'Number of categories {len(item_categories)}')

item_categories.head(4)
print(f'Number of products: {len(items)}')

items.head()
x = items['item_category_id'].value_counts()

fig_dims = (20, 7)

fig, ax = plt.subplots(figsize=fig_dims)

sns.barplot(x.index, x)
item_categories.iloc[[40,55,37,31,72]]
print(f'Number of different shops: {len(shops)}')

shops.head()
print(f'Rows test set: {len(test)}')

test.head()
n_p_t = len(test['item_id'].unique())

print(f'Number of different products in test: {n_p_t}')
x = test['item_id'].value_counts()[test['item_id'].value_counts() > 30]

print(x)
n_shops = len(test['shop_id'].unique())

n_products = len(test['item_id'].unique())

print(f'Number of different shops in test set: {n_shops}')

print(f'Number of different products in test set: {n_products}')
x = test['shop_id'].value_counts()

fig_dims = (20, 7)

fig, ax = plt.subplots(figsize=fig_dims)

sns.barplot(x.index, x)
print(f'Number of training samples: {len(sales_train)}')

sales_train.head()
sales_train.tail()
sales_train['item_price'].describe()
sales_train[sales_train['item_price'] <= 0]
id = sales_train[sales_train.item_price <= 0].index

sales_train = sales_train.drop(id)
sales_train.isnull().sum(axis = 0)
x = sales_train['shop_id'].value_counts()

fig_dims = (20, 7)

fig, ax = plt.subplots(figsize=fig_dims)

sns.barplot(x.index, x)
x = test['shop_id'].value_counts()

fig_dims = (20, 7)

fig, ax = plt.subplots(figsize=fig_dims)

sns.barplot(x.index, x)
id = sales_train[sales_train.shop_id.isin([0,1,8,9,11,20,32,33,40])].index

sales_train.drop(id, inplace=True)
sales_train
x = sales_train['shop_id'].value_counts()

fig_dims = (20, 7)

fig, ax = plt.subplots(figsize=fig_dims)

sns.barplot(x.index, x)
x = sales_train['item_id'].value_counts()[sales_train['item_id'].value_counts() > 1000]

fig_dims = (20, 7)

fig, ax = plt.subplots(figsize=fig_dims)

sns.barplot(x.index, x)
p_id = sales_train['item_id'].value_counts().idxmax()

item_prices_daily = sales_train[sales_train['item_id'] == p_id].sort_values(by='date')['item_price']

sns.lineplot(data=item_prices_daily)
item_prices_daily.value_counts()
sales_train['item_cnt_day'].describe()
sales_train.item_cnt_day.value_counts()
len(sales_train[sales_train.item_cnt_day <= 0])/len(sales_train)
sales_train.drop(sales_train[sales_train.item_cnt_day < 0].index, inplace=True)
u_train_id = sales_train['item_id'].unique()

u_test_id = test['item_id'].unique()

5100 - len(np.intersect1d(u_test_id,u_train_id))
sales_train
# Item per shop and month.

ipsm = sales_train[['shop_id','date_block_num','item_id','item_cnt_day','item_price']].groupby(['date_block_num','shop_id','item_id'],as_index=False).sum()
ipsm.rename(columns={'item_cnt_day':'item_cnt_month'}, inplace=True)

ipsm.sample(100)
ipsm = ipsm.merge(items,how='left',on='item_id')
ipsm.drop('item_name',axis=1,inplace=True)
ipsm.sample(1000)
ipsm.sort_values('date_block_num',inplace=True)
from fastai.tabular import *
valid_idx = range(len(ipsm)-160_000, len(ipsm))

valid_idx
procs = [FillMissing, Categorify, Normalize]
dep_var = 'item_cnt_month'

cat_names = ['date_block_num','shop_id','item_id','item_category_id']

cont_names = ['item_price']
data = (TabularList.from_df(ipsm, path='.', cat_names=cat_names, cont_names=cont_names, procs=procs)

                   .split_by_idx(valid_idx)

                   .label_from_df(cols=dep_var, label_cls=FloatList)

                   .databunch())
data
(cat_x,cont_x),y = next(iter(data.train_dl))

for o in (cat_x, cont_x, y): print(to_np(o[:5]))
learn = tabular_learner(data, layers=[200,100], metrics=[rmse])
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(15)
from sklearn.preprocessing import StandardScaler, LabelEncoder
labelencoder = LabelEncoder()



ipsm['date_block_num'] = labelencoder.fit_transform(ipsm['date_block_num'])
ipsm
from xgboost import XGBRegressor
y = ipsm['item_cnt_month']

X = ipsm.drop(['item_cnt_month'], axis=1)



valid_sz = int(len(X) * 0.1)



X_train = np.array(X)[valid_sz:]

y_train = np.array(y)[valid_sz:]



X_valid = np.array(X)[:valid_sz]

y_valid = np.array(y)[:valid_sz]
X
X_train.shape
XGB = XGBRegressor(max_depth=3,learning_rate=0.1,n_estimators=1000,reg_alpha=0.001,reg_lambda=0.000001,n_jobs=-1,min_child_weight=3)

XGB.fit(X_train,y_train, verbose=True)
XGB.score(X_valid, y_valid)
XGB.score(X_train, y_train)
y_hat = XGB.predict(X_valid)

np.sqrt(sum((y_hat - y_valid)**2)*1/len(y_hat))
test = test.merge(items,how='left',on='item_id')

test.drop(columns=['item_name'], axis=1, inplace=True)
test['date_block_num'] = 34
test
X_test = np.array(test)

y_hat = xg_reg.predict(test)
preds = []

for i,y in enumerate(y_hat):

    preds.append([i, y])
preds
submission = pd.DataFrame(preds, columns=['ID', 'item_cnt_month'])
submission = submission.set_index('ID')

submission.head()
submission.to_csv('submission.csv')