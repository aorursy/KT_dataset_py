import numpy as np 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys
from datetime import datetime
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout
from keras.models import load_model, Model
from fbprophet import Prophet
df = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv', parse_dates = ['date'], infer_datetime_format = True, dayfirst = True)
df.tail()
df.fillna(0,inplace = True)
df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
df['week'] = df['date'].dt.week
plt.rcParams['figure.figsize'] = (16, 7)
sns.countplot(df['year'], palette = 'dark')
plt.title('En yoğun yıl', fontsize = 30)
plt.xlabel('Yıl', fontsize = 10)
plt.ylabel('Yoğunluk', fontsize = 10)

plt.show()
plt.rcParams['figure.figsize'] = (15, 7)
sns.countplot(df['month'], palette = 'bright')
plt.title('En Yoğun Aylar', fontsize = 30)
plt.xlabel('Ay', fontsize = 15)
plt.ylabel('Yoğunluk', fontsize = 15)
plt.show()
items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')
shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')
item_categories = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')
sales_train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv',parse_dates = ['date'], infer_datetime_format = True, dayfirst = True)
test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')
train = sales_train.join(items, on='item_id', rsuffix='_').join(shops, on='shop_id', rsuffix='_').join(item_categories, on='item_category_id', rsuffix='_').drop(['item_id_', 'shop_id_', 'item_category_id_'], axis=1)
train
test_shop_ids = test['shop_id'].unique()
test_item_ids = test['item_id'].unique()
lk_train = train[train['shop_id'].isin(test_shop_ids)]
lk_train = lk_train[lk_train['item_id'].isin(test_item_ids)]
print(lk_train.shape)
lk_train
sales_train = sales_train.query('item_price > 0')
sales_train = sales_train[sales_train['shop_id'].isin(test['shop_id'].unique())]
sales_train = sales_train[sales_train['item_id'].isin(test['item_id'].unique())]
sales_train = sales_train.query('item_cnt_day >= 0')
sales_train.shape
monthly_sales=sales_train.groupby(["date_block_num","shop_id","item_id"])["date_block_num","date","item_price","item_cnt_day"].agg({"date_block_num":'mean',"date":["min",'max'],"item_price":"mean","item_cnt_day":"sum"})
monthly_sales.head()
sales_data_flat = monthly_sales.item_cnt_day.apply(list).reset_index()
sales_data_flat = pd.merge(test,sales_data_flat,on = ['item_id','shop_id'],how = 'left')
sales_data_flat.fillna(0,inplace = True)
sales_data_flat.drop(['shop_id','item_id'],inplace = True, axis = 1)
sales_data_flat.head(10)
sales = sales_data_flat.pivot_table(index='ID', columns='date_block_num',fill_value = 0,aggfunc='sum' )
sales.head(10)
X_train = np.expand_dims(sales.values[:,:-1],axis = 2)
y_train = sales.values[:,-1:]
X_test = np.expand_dims(sales.values[:,1:],axis = 2)
model = Sequential()
model.add(LSTM(units = 15,input_shape = (33,1)))
model.add(Dropout(0.01, input_shape=(60,)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss = 'mse',optimizer = 'adam', metrics=['mse', 'mae'])
model.fit(X_train,y_train,batch_size = len(X_train),epochs = 10)
output = model.predict(X_test)
result = pd.DataFrame({'ID':test['ID'],'item_cnt_month':output.ravel()})
group=sales_train.groupby(["date_block_num"])["item_cnt_day"].sum()
group.index=pd.date_range(start = '2013-01-01',end='2015-10-01', freq = 'MS')
group=group.reset_index()
group.rename(columns={'index': 'date'}, inplace=True)
group.columns=['ds','y']
model = Prophet( yearly_seasonality=True) 
model.fit(group)
future = model.make_future_dataframe(periods = 10, freq = 'MS')  
predict = model.predict(future)
model.plot(predict)