# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout
from keras.models import load_model, Model
# Veri kümelerinin okunması
items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')
sales_train = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')
item_categories = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')
test = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
shops = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')
sample_submission = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv')
#Veri Okuma Kısmı
item_categories.head()
item_categories['item_category_name'].count()
item_categories.isnull().sum()

x=items.groupby(['item_category_id']).count()
x=x.sort_values(by='item_id',ascending=False)
x=x.iloc[0:10].reset_index()

plt.figure(figsize=(8,4))
ax= sns.kdeplot(x.item_category_id, x.item_id, alpha=0.8)
plt.title("Items per Category")
plt.ylabel('# of items', fontsize=12)
plt.xlabel('Category', fontsize=12)
plt.show()


egitim_datası = sales_train.query('item_price > 0')

egitim_datası = egitim_datası[egitim_datası['shop_id'].isin(test['shop_id'].unique())]

egitim_datası = egitim_datası[egitim_datası['item_id'].isin(test['item_id'].unique())]

egitim_datası = egitim_datası.query('item_price < 60000')

egitim_datası['year'] = pd.to_datetime(egitim_datası['date']).dt.strftime('%Y')

egitim_datası['month'] = egitim_datası.date.apply(lambda x: datetime.strptime(x,'%d.%m.%Y').strftime('%m')) 

egitim_datası.head(2)

cleaned = pd.DataFrame(egitim_datası.groupby(['year','month'])['item_cnt_day'].sum().reset_index())






monthly_sales=egitim_datası.groupby(["date_block_num","shop_id","item_id"])[
"date_block_num","date","item_price","item_cnt_day"].agg({"date_block_num":'mean',"date":["min",'max'],"item_price":"mean","item_cnt_day":"sum"})

monthly_sales.head()
sales_data_flat = monthly_sales.item_cnt_day.apply(list).reset_index()


sales_data_flat = pd.merge(test,sales_data_flat,on = ['item_id','shop_id'],how = 'left')
sales_data_flat.head(10)
sales_data_flat.fillna(0,inplace = True)
sales_data_flat.drop(['shop_id','item_id'],inplace = True, axis = 1)

sales = sales_data_flat.pivot_table(index='ID', columns='date_block_num',fill_value = 0,aggfunc='sum' )
sales.head(10)


X_train = np.expand_dims(sales.values[:,:-1],axis = 2)

y_train = sales.values[:,-1:]

X_test = np.expand_dims(sales.values[:,1:],axis = 2)

print(X_train.shape,y_train.shape,X_test.shape)
sales_model = Sequential()
sales_model.add(LSTM(units = 64,input_shape = (33,1)))
sales_model.add(Dropout(0.5))
sales_model.add(Dense(1))

sales_model.compile(loss = 'mse',optimizer = 'adam', metrics = ['mean_squared_error'])
sales_model.summary()
sales_model.fit(X_train,y_train,batch_size = 4096,epochs = 8)
submission_output = sales_model.predict(X_test)

submission = pd.DataFrame({'ID':test['ID'],'item_cnt_month':submission_output.ravel()})
submission.head()

submission.to_csv('submission.csv',index = False)
ts=sales_train.groupby(["date_block_num"])["item_cnt_day"].sum()
ts.index=pd.date_range(start = '2013-01-01',end='2015-10-01', freq = 'MS')
ts=ts.reset_index()
ts.head()
from fbprophet import Prophet
ts.columns=['ds','y']
model = Prophet( yearly_seasonality=True)
model.fit(ts)
future = model.make_future_dataframe(periods = 3, freq = 'MS')  
forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()