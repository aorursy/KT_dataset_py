import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout
from keras.models import load_model, Model
item_categories = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')
items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')
shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')
sample_submission = pd.read_csv('../input/competitive-data-science-predict-future-sales/sample_submission.csv')
test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')
sales_train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')

item_categories.head()

item_categories.isnull().sum()
items.head()
items.item_id.nunique()
x=items.groupby(['item_category_id']).count()
x=x.sort_values(by='item_id',ascending=False)
x=x.iloc[0:10].reset_index()
plt.figure(figsize=(8,4))
ax= sns.barplot(x.item_category_id, x.item_id, alpha=0.8)
plt.title("Items per Category")
plt.ylabel('# of items', fontsize=12)
plt.xlabel('Category', fontsize=12)
plt.show()
shops.head()
sample_submission.head()
test.head()
sales_train.head()
sales_train.describe()
sales_train = sales_train.query('item_price > 0')

# test.csv'de bulunan shop ve item'lar alınmıştır.
sales_train = sales_train[sales_train['shop_id'].isin(test['shop_id'].unique())]
sales_train = sales_train[sales_train['item_id'].isin(test['item_id'].unique())]

sales_train = sales_train.query('item_cnt_day >= 0 and item_price < 75000')
sales_train['year'] = pd.to_datetime(sales_train['date']).dt.strftime('%Y')
sales_train['month'] = pd.to_datetime(sales_train['date']).dt.strftime('%m')
sales_train.head(5)
cleaned = pd.DataFrame(sales_train.groupby(['year','month'])['item_cnt_day'].sum().reset_index())
sns.pointplot(x='month', y='item_cnt_day', hue='year', data=cleaned)

monthly_sales=sales_train.groupby(["date_block_num","shop_id","item_id"])["date_block_num","date","item_price","item_cnt_day"].agg({"date_block_num":'mean',"date":["min",'max'],"item_price":"mean","item_cnt_day":"sum"})

monthly_sales.head()

sales_data_flat = monthly_sales.item_cnt_day.apply(list).reset_index()
#Sadece geçerli test verileri saklanır.

sales_data_flat.head(10)
sales_data_flat = pd.merge(test,sales_data_flat,on = ['item_id','shop_id'],how = 'left')
sales_data_flat.head(10)



sales_data_flat.fillna(0,inplace = True)
sales_data_flat.drop(['shop_id','item_id'],inplace = True, axis = 1)
sales_data_flat.head(10)
sales = sales_data_flat.pivot_table(index='ID', columns='date_block_num',fill_value = 0,aggfunc='sum' )
sales.head(10)
# Son sütun hariç hepsini X e atıyoruz.
X_train = np.expand_dims(sales.values[:,:-1],axis = 2)
# Son sütunu y ye atıyoruz (tahmin edilecek olan değer)
y_train = sales.values[:,-1:]

# test için oluşturulan X
X_test = np.expand_dims(sales.values[:,1:],axis = 2)

 
print(X_train.shape,y_train.shape,X_test.shape)

sales_model = Sequential()
sales_model.add(LSTM(units = 64,input_shape = (33,1)))
sales_model.add(Dropout(0.5))
sales_model.add(Dense(1))

sales_model.compile(loss = 'mse',optimizer = 'adam', metrics = ['mean_squared_error'])
sales_model.summary()
sales_model.fit(X_train,y_train,batch_size = 4096,epochs = 6)
submission_output = sales_model.predict(X_test)
# Beklenen sütunlarla df oluşturuldu. 
submission = pd.DataFrame({'ID':test['ID'],'item_cnt_month':submission_output.ravel()})
submission.head()
# Oluşan df'den csv dosyası oluşturuldu.
submission.to_csv('submission.csv',index = False)

ts=sales_train.groupby(["date_block_num"])["item_cnt_day"].sum()
ts.index=pd.date_range(start = '2013-01-01',end='2015-10-01', freq = 'MS')
ts=ts.reset_index()
ts.head()
from fbprophet import Prophet
# date sütunu = DS , value sütunu = Y
ts.columns=['ds','y']
model = Prophet(yearly_seasonality=True) 
model.fit(ts)
# Gelecek 5 ay için tahmin
future = model.make_future_dataframe(periods = 5, freq = 'MS')  

forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
model.plot_components(forecast)
