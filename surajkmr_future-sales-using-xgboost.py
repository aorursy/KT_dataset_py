import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
train_data = pd.read_csv("../input/sales_train.csv")

test_data =  pd.read_csv("../input/test.csv")

items = pd.read_csv("../input/items.csv")

categories = pd.read_csv("../input/item_categories.csv") 

shops_data = pd.read_csv("../input/shops.csv")

train_data.head()
median = train_data[(train_data.shop_id==32)&(train_data.item_id==2973)&(train_data.date_block_num==4)&(train_data.item_price>0)].item_price.median()

train_data.loc[train_data.item_price<0, 'item_price'] = median

train_data['date'] = pd.to_datetime(train_data.date,format="%d.%m.%Y")



cat_list = list(categories.item_category_name)

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



categories['cats'] = cat_list

categories.head()
(items.item_category_id==25).sum()
train_data.info()

print ( sorted(train_data["shop_id"].unique()) )

print (sorted(test_data["shop_id"].unique()))
train_data.info()

print (train_data.shape)
subset = ['date', 'date_block_num', 'shop_id', 'item_id','item_cnt_day']

train_data.drop_duplicates(keep="first", subset = subset, inplace=True)

train_data.shape
plt.figure(figsize=(10,6))

train_data.hist(figsize=(16, 16), bins=50, xlabelsize=8, ylabelsize=8)
train_data.groupby('date_block_num')['item_cnt_day'].sum().plot.line()

plt.title("no of total products sold in each month", fontsize = 14)

plt.xlabel('date_block_num', fontsize=12)

plt.ylabel('# of products', fontsize=12)
train_data.groupby('shop_id')['item_cnt_day'].sum().plot.line()

plt.title("no of total products sold of different shop", fontsize=14)

plt.xlabel('shop_id', fontsize=12)

plt.ylabel('# of products', fontsize=12)
#items.sample(5)

x=items.groupby(['item_category_id']).count()

x=x.sort_values(by='item_id',ascending=False)

x=x.iloc[0:20].reset_index()

plt.figure(figsize=(10,6))

ax= sns.barplot(x.item_category_id, x.item_id, alpha=0.8)

plt.title("Items per Category")

plt.ylabel('No. of items', fontsize=12)

plt.xlabel('Category', fontsize=12)

plt.show()
train_data['day_of_week'] = train_data['date'].dt.day_name()

plt.figure(figsize=(10,6))

day_wise=train_data.groupby('day_of_week')['item_cnt_day'].sum().plot.bar()

plt.title("sale based on week days")

plt.ylabel('No. of items', fontsize=12)

plt.xlabel('days', fontsize=12)
all_data = pd.merge(train_data, items, how='left', on=['item_id','item_id'])

all_data = pd.merge(all_data, categories, how = "left", on = ['item_category_id','item_category_id'])

all_data = pd.merge(all_data, shops_data, how = "left", on = ['shop_id','shop_id'])

all_data['month'] = all_data['date'].dt.month

all_data['year'] = all_data['date'].dt.year

all_data['revenue'] =all_data.item_price * all_data.item_cnt_day

all_data.dtypes
all_data.head(3)
train_df = all_data[['year', 'month', 'day_of_week', 'date_block_num', 'shop_id', 'item_id', 'item_category_id', 'item_cnt_day', 'revenue']]

print(train_df.shape)
shops_test = test_data.shop_id.unique()

items_test = test_data.item_id.unique()

train_df = train_df[train_df.shop_id.isin(shops_test) & train_df.item_id.isin(items_test) ]

print('train_df:', train_df.shape)
train_df[train_df.item_cnt_day < 0]
test_data.head()
print(train_df.tail())

pivoted_train = train_df.pivot_table(index=['shop_id','item_id'], columns='date_block_num', values='item_cnt_day',aggfunc='sum').fillna(0.0)

pivoted_train.tail()
data_clean = pivoted_train.reset_index()

data_clean['shop_id']= data_clean.shop_id.astype('str')

data_clean['item_id']= data_clean.item_id.astype('str')

item_to_cat = items.merge(categories[['item_category_id','cats']], how="inner", on="item_category_id")[['item_id','cats']]

item_to_cat[['item_id']] = item_to_cat.item_id.astype('str')

data_clean = data_clean.merge(item_to_cat, how="inner", on="item_id")

data_clean.head(10)
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

data_clean[['cats']] = le.fit_transform(data_clean.cats)

data_clean = data_clean[['shop_id', 'item_id', 'cats'] + list(range(34))]

data_clean.head()
import xgboost as xgb

X_train = data_clean.iloc[:, :-1].values

y_train = data_clean.iloc[:, -1].values

progress = dict()

param = {'max_depth':40,'min_child_weight':0.5,'eta':0.3,'num_round':100,'seed':0,'eval_metric':'rmse', 'early_stopping_rounds':1000 }

xgbtrain = xgb.DMatrix(X_train, y_train)

watchlist  = [(xgbtrain,'train-rmse')]

bst = xgb.train(param, xgbtrain)
from sklearn.metrics import mean_squared_error 

preds = bst.predict(xgb.DMatrix(X_train))

rmse = np.sqrt(mean_squared_error(preds, y_train))

print(rmse)
test_data['shop_id']= test_data.shop_id.astype('str')

test_data['item_id']= test_data.item_id.astype('str')

test_data = test_data.merge(data_clean, how = "left", on = ["shop_id", "item_id"]).fillna(0.0)

test_data.head()
d = dict(zip(test_data.columns[4:], list(np.array(list(test_data.columns[4:])) - 1)))

test_data  = test_data.rename(d, axis = 1)

X_test = test_data.drop(['ID', -1], axis=1).values

preds = bst.predict(xgb.DMatrix(X_test))

print(preds.shape)
predict_sale = pd.DataFrame({'ID':test_data.ID, 'item_cnt_month': preds.clip(0. ,20.)})

predict_sale.to_csv('submission_sale.csv',index=False)