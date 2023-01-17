import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
submission=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv')
shops=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')
items=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')
item_cats=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')
train=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')
test=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
test.head()
items.head()
item_cats.head()
shops.head()
submission.head()
train.head()
print("shape of train:",train.shape)
print("shape of test:",test.shape)
print("shape of submission:",submission.shape)
print("shape of items:",items.shape)
print("shape of item_cats:",item_cats.shape)
print("shape of shops:",shops.shape)
# drop duplicates 
subset = ['date','date_block_num','shop_id','item_id','item_cnt_day'] 
print(train.duplicated(subset=subset).value_counts()) 
train.drop_duplicates(subset=subset, inplace=True)
print("shape of train:",train.shape)
# drop shops&items not in test data 
test_shops = test.shop_id.unique() 
test_items = test.item_id.unique() 
train = train[train.shop_id.isin(test_shops)] 
train = train[train.item_id.isin(test_items)] 
print("shape of train:",train.shape)
fig = plt.figure(figsize=(18,9))
plt.subplots_adjust(hspace=.5)

plt.subplot2grid((3,3), (0,0), colspan = 3)
train['shop_id'].value_counts(normalize=True).plot(kind='bar', alpha=0.7)
plt.title('Shop ID Values in the Training Set (Normalized)')

plt.subplot2grid((3,3), (1,0))
train['item_id'].plot(kind='hist', alpha=0.7)
plt.title('Item ID Histogram')

plt.subplot2grid((3,3), (1,1))
train['item_price'].plot(kind='hist', alpha=0.7, color='orange')
plt.title('Item Price Histogram')

plt.subplot2grid((3,3), (1,2))
train['item_cnt_day'].plot(kind='hist', alpha=0.7, color='green')
plt.title('Item Count Day Histogram')

plt.subplot2grid((3,3), (2,0), colspan = 3)
train['date_block_num'].value_counts(normalize=True).plot(kind='bar', alpha=0.7)
plt.title('Month (date_block_num) Values in the Training Set (Normalized)')

plt.show()
fig = plt.figure(figsize=(10,5))

plt.subplot2grid((1,2), (0,0))
train['item_price'].plot(kind='box')
plt.title('Item price')

plt.subplot2grid((1,2), (0,1))
train['item_cnt_day'].plot(kind='box')
plt.title('item_cnt_day')

plt.show()
train['month']=train['date_block_num']
train['month']+=1
train['month']%=12
print('shape of train',train.shape)
train['month']
#season
train['season']=train['month']
train['season']//=3
train['season']=train['season'].replace(0,4)
#area
train=pd.merge(train,shops,how='left',on='shop_id')
area = train['shop_name'].apply(lambda x: str.replace(x, '!', '')).apply(lambda x: x.split(' ')[0]) 
train['area'] = pd.Categorical(area).codes 
train.head()
train.shape
from itertools import product

block_shop_combi = pd.DataFrame(list(product(np.arange(34), test_shops)), columns=['date_block_num','shop_id']) 
shop_item_combi = pd.DataFrame(list(product(test_shops, test_items)), columns=['shop_id','item_id']) 
all_combi = pd.merge(block_shop_combi, shop_item_combi, on=['shop_id'], how='inner')
train_base = pd.merge(all_combi, train, on=['date_block_num','shop_id','item_id'], how='left') 
train_base['item_cnt_day'].fillna(0, inplace=True) 
train_grp = train_base.groupby(['date_block_num','shop_id','item_id'])
train_month = pd.DataFrame(train_grp.agg({'item_cnt_day':['sum','count']})).reset_index() 
train_month.columns = ['date_block_num','shop_id','item_id','item_cnt','item_order']
train_month.head()
grp = train_month.groupby(['shop_id', 'item_id']) 
train_shop = grp.agg({'item_cnt':['mean','median','std'],'item_order':'mean'}).reset_index() 
train_shop.columns = ['shop_id','item_id','cnt_mean_shop','cnt_med_shop','cnt_std_shop','order_mean_shop'] 
print(train_shop[['cnt_mean_shop','cnt_med_shop','cnt_std_shop']].describe())
train_shop.head()
price_max = train.groupby(['item_id']).max()['item_price'].reset_index()
price_max.rename(columns={'item_price':'item_max_price'}, inplace=True)
price_max.head()
price = train.groupby(['item_id']).mean()['item_price'].reset_index()
price.rename(columns={'item_price':'item_mean_price'}, inplace=True)
price.head()
train_price_dc = pd.merge(price, price_max, on=['item_id'], how='left') 
train_price_dc['discount'] = 1 - (train_price_dc['item_mean_price'] / train_price_dc['item_max_price']) 
train_price_dc.drop('item_max_price', axis=1, inplace=True) 
train_price_dc.head()
train=pd.merge(train,train_price_dc,how='left',on='item_id')
train
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
X = train.iloc[:, 0:12]
Y = train.iloc[:, 12]
x_train, x_test, y_train, y_test = train_test_split(X, Y, stratify = Y, random_state = 0)
svc = SVC(C = 1000, gamma = 0.00001)
svc.fit(x_train, y_train)
print("prediction :", svc.predict(x_test))
print("train accuracy :", svc.score(x_train, y_train))
print("test accuracy :", svc.score(x_test, y_test))
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=10, random_state=0)
forest.fit(X_train, y_train)

print("prediction :", forest.predict(X_test))
print("train accuracy :", forest.score(X_train, y_train))
print("test accuracy :", forest.score(X_test, y_test))
from sklearn.ensemble import GradientBoostingClassifier

gbrt = GradientBoostingClassifier(random_state=0,max_depth=1)

gbrt.fit(X_train,y_train)


print("train accuracy :".format(gbrt.score(X_train,y_train)))
print("test accuracy :".format(gbrt.score(X_test,y_test)))