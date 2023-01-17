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
import pandas as pd
import datetime
items= pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/items.csv")
shops= pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/shops.csv")
sales_train= pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv")
test= pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/test.csv")
sample_submission= pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv")
item_categories= pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv")
shops.head()
sales_train.head()

shops.loc[shops["shop_name"] == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'
shops["city"]=shops["shop_name"].str.split(' ').map(lambda x: x[0])
shops["name_of_shop"] = shops["shop_name"].str.split(' ').map(lambda x: x[1])
shops.loc[shops["city"] =="!Якутск","city"] = "Якутск"


shops.loc[shops["name_of_shop"] =='"Распродажа"',"name_of_shop"] = "other"
shops.loc[shops["name_of_shop"] =="(Плехановская,","name_of_shop"] = "other"
shops.loc[shops["name_of_shop"] =="МТРЦ","name_of_shop"] = "ТРЦ"
from sklearn.preprocessing import LabelEncoder
shops["name_of_shop"] = LabelEncoder().fit_transform(shops["name_of_shop"])
shops["city"] = LabelEncoder().fit_transform(shops["city"])
shops


sales_train.info()
test.info()
sales_train = sales_train.merge(items, on = "item_id" )
#sales_train = sales_train.merge(shops, on = "shop_id" )
sales_train.head()
sales_train = sales_train.drop(["item_name","item_category_id","item_price"], axis = 1)
sales_train["date"] = pd.to_datetime(sales_train["date"])
sales_train.head()
sales_train["date_block_num"].unique()
sales_train_monthly =pd.DataFrame(sales_train.sort_values("date").groupby(["date_block_num","shop_id","item_id"])["item_cnt_day"].agg('sum').reset_index())
sales_train_monthly.describe().T
sales_train_monthly_pivot = sales_train_monthly.pivot_table(index = ['shop_id','item_id'],values = ['item_cnt_day'],columns = ['date_block_num']).reset_index()
sales_train_monthly_pivot
sales_train_monthly_pivot= pd.merge(test,sales_train_monthly_pivot,on = ['item_id','shop_id'],how = 'left')
sales_train_monthly_pivot.fillna(0,inplace = True)
sales_train_monthly_pivot.head(10)

from sklearn.model_selection import train_test_split
#sales_train_monthly_pivot = sales_train_monthly_pivot.drop(["shop_id", "item_id", "ID"], axis = 1)

sales_train_monthly_pivot.head()
#proba
X= sales_train_monthly_pivot.iloc[:,:-1]
y= sales_train_monthly_pivot.iloc[:,-1]
y =y.values.reshape(-1,1)
print(X.shape, y.shape)
#proba2
X_train= sales_train_monthly_pivot.iloc[:,:-1]
y_train= sales_train_monthly_pivot.iloc[:,-1]
X_test = sales_train_monthly_pivot.iloc[:,1:]

y_train =y_train.values.reshape(-1,1)
print(X_train.shape,y_train.shape,X_test.shape)

from sklearn.preprocessing import MinMaxScaler
X_train.info()
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
X_train.shape
X_test.shape
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
print(X_test.shape, X_test.shape)
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout
my_model = Sequential()
my_model.add(LSTM(units = 10,return_sequences = True,input_shape = (36,1)))
my_model.add(Dropout(0.2))
my_model.add(LSTM(units = 10,return_sequences = True))
my_model.add(Dropout(0.2))
my_model.add(LSTM(units = 10,return_sequences = True))
my_model.add(Dropout(0.2))
my_model.add(LSTM(units = 10,return_sequences = True))
my_model.add(Dropout(0.2))
my_model.add(LSTM(units = 10,return_sequences = True))
my_model.add(Dropout(0.2))
my_model.add(LSTM(units = 10,return_sequences = False))
my_model.add(Dropout(0.2))
my_model.add(Dense(1))




my_model.compile(loss = 'mse',optimizer = 'adam', metrics = ['mean_squared_error'])
my_model.summary()
my_model.fit(X_train,y_train,batch_size = 12500,epochs = 20)
from tensorflow.keras.models import load_model
losses = pd.DataFrame(my_model.history.history)
losses.plot()
"""#test podatke vzemen
X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))"""
X_test.shape
submission_pfs = my_model.predict(X_test)
submission_pfs = submission_pfs.clip(0,20)

submission = pd.DataFrame({'ID':test['ID'],'item_cnt_month':submission_pfs.ravel()})
submission.to_csv('forceste_price_36.csv', index = False)
