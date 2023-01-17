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
import pandas as pd

path = '/kaggle/input/competitive-data-science-predict-future-sales/'

items = pd.read_csv(path+'/items.csv')

items_cats = pd.read_csv(path+'/item_categories.csv')

shops = pd.read_csv(path+'/shops.csv')

sales = pd.read_csv(path+'/sales_train.csv')

test = pd.read_csv(path+'/test.csv')

submission = pd.read_csv(path+'/sample_submission.csv')

print('Dataset Loading Successful')
print(items.info())

print(sales.info())
print('Items')

print(items.head())

print('\nItem Categories')

print(items_cats.tail(2))

print('\nshops')

print(shops.sample(n=3))

print('\n Training Dataset')

print(sales.sample(n=3,random_state=1))

print('\n Test Dataset')

print(test.sample(n=3,random_state=1))
from datetime import datetime

sales['year']=pd.to_datetime(sales['date']).dt.strftime('%Y')

sales['month']=pd.to_datetime(sales['date']).dt.strftime('%m')

sales['day']=pd.to_datetime(sales['date']).dt.strftime('%d')

sales.head()
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline 



grouped = pd.DataFrame(sales.groupby(['year','month'])

	['item_cnt_day'].sum().reset_index())

sns.barplot(x='month',y='item_cnt_day',hue='year',data = grouped)



import seaborn as sns

%matplotlib inline 





sns.boxplot(x='month',y='item_cnt_day',data=grouped)

sns.pointplot(x='month',y='item_cnt_day',hue='year',data=grouped)
grouped_price = pd.DataFrame(sales.groupby(['year','month'])

	['item_price'].mean().reset_index())

sns.barplot(x='month',y='item_price',hue='year',data=grouped_price)
sns.pointplot(x='month',y='item_price',hue='year',data=grouped_price)
sns.violinplot(x='month',y='item_price',hue='year',data=grouped_price)
sns.jointplot(x='item_price',y='item_cnt_day',data=sales,height=10)

plt.show()
print('Data set size before modification::',sales.shape)

sales=sales.query('item_price > 0')

print('Data set size after modification',sales.shape)

sales = sales.query('item_cnt_day >= 0 and item_cnt_day<=500 and item_price < 60000')

print(' Data shape prior modification:',sales.shape)

sns.jointplot(x='item_cnt_day',y='item_price',data=sales,height=8)



plt.show()





cleaned = pd.DataFrame(sales.groupby(['year','month'])['item_cnt_day'].sum().reset_index())

sns.violinplot(x='month',y='item_cnt_day',hue='year',data=cleaned)
#Selecting only relevant features

monthly_sales=sales.groupby(['date_block_num','shop_id','item_id'])['date_block_num',

                                                                   'date','item_price','item_cnt_day'].agg({'date_block_num':'mean','date':['min','max'],'item_price':'mean','item_cnt_day':'sum'})

monthly_sales.head()
sales_data_flat = monthly_sales.item_cnt_day.apply(list).reset_index()

#keeping only the test data of valid

sales_data_flat = pd.merge(test,sales_data_flat,on = ['item_id','shop_id'],how = 'left')

#filling na with zeroes

sales_data_flat.fillna(0,inplace = True)

sales_data_flat.drop(['shop_id','item_id'],inplace = True , axis =1)

sales_data_flat.head()
#Creating the pivot table

pivoted_sales = sales_data_flat.pivot_table(index='ID',columns='date_block_num',fill_value=0,aggfunc = 'sum')

pivoted_sales.head()
X_train = np.expand_dims(pivoted_sales.values[:,:-1],axis=2)

#The last column is our prediction

y_train = pivoted_sales.values[:,-1:]

X_test = np.expand_dims(pivoted_sales.values[:,1:],axis=2)

print(X_train.shape,y_train.shape,X_test.shape)
from keras.models import Sequential

from keras.layers import LSTM, Dense,Dropout

from keras.models import load_model, Model



sales_model = Sequential()

sales_model.add(LSTM(units = 64 , input_shape = (33,1),activation='relu'))

#sales_model.add(LSTM(units = 64, activation = 'relu'))

sales_model.add(Dropout(0.7))

sales_model.add(Dense(1))

sales_model.compile(loss='mse',optimizer = 'adam',metrics=['mean_squared_error'])

sales_model.summary()
sales_model.fit(X_train,y_train,batch_size=4096,epochs=50)
submission_output = sales_model.predict(X_test)

submission = pd.DataFrame({'ID':test['ID'],'item_cnt_month':submission_output.ravel()})

submission.to_csv('submission_stacked.csv',index = False)

submission.head()