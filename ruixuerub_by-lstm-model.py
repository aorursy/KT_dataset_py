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
import numpy as np

import pandas as pd 

import os
sales_data = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv', dtype={'date':'str', 'date_block_num':'int64', 'shop_id':'int64', 'item_id':'int64', 'item_price':'float64', 'item_cnt_day':'float64'})

item_cat = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv', dtype={'item_category_name':'str', 'item_category_id':'int64'})

items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv', dtype={'item_name':'str', 'item_id':'int64', 'item_category_id':'int64'})

shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv', dtype={'shop_name':'str', 'shop_id':'int64'})

test_data = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv', dtype={'ID':'int64', 'shop_id':'int64', 'item_id':'int64'})
print('number of shops: ',shops['shop_id'].max())

print('number of products: ',items['item_id'].max())

print('number of product categories: ',item_cat['item_category_id'].max())

print('number of month: ',sales_data['date_block_num'].max())

print('number of ID: ', test_data['ID'].max())
sales_data.head()
item_cat.head()
items.head()
shops.head()
test_data.head()
sales_data['date'] = pd.to_datetime(sales_data['date'],format = '%d.%m.%Y')
dataset = sales_data.pivot_table(index = ['shop_id','item_id'],values = ['item_cnt_day'],columns = ['date_block_num'],fill_value = 0,aggfunc='sum')
dataset = sales_data.pivot_table(index = ['shop_id','item_id'],values = ['item_cnt_day'],columns = ['date_block_num'],fill_value = 0,aggfunc='sum')
# lets reset our indices, so that data should be in way we can easily manipulate

dataset.reset_index(inplace = True)
dataset.head()
dataset = pd.merge(test_data,dataset,on = ['item_id','shop_id'],how = 'left')

dataset.head()
# lets fill all NaN values with 0

dataset.fillna(0,inplace = True)

#dataset.fillna(dataset['item_cnt_day'].mean,inplace = True)

# lets check our data now 

dataset.head()
#dataset = pd.merge(test_data,dataset,on = ['item_id','shop_id'],how = 'left')

#dataset.head()
# we will drop shop_id and item_id because we do not need them

# we are teaching our model how to generate the next sequence 

dataset.drop(['shop_id','item_id','ID'],inplace = True, axis = 1)

dataset.head()
# X we will keep all columns execpt the last one 

X_train = np.expand_dims(dataset.values[:,:-1],axis = 2)

# the last column is our label

y_train = dataset.values[:,-1:]



# for test we keep all the columns execpt the first one

X_test = np.expand_dims(dataset.values[:,1:],axis = 2)



# lets have a look on the shape 

print(X_train.shape,y_train.shape,X_test.shape)
# importing libraries required for our model

from keras.models import Sequential

from keras.layers import LSTM,Dense,Dropout,Activation
from keras import backend as K



# our defining our model 

my_model = Sequential()

my_model.add(LSTM(units = 64,input_shape = (33,1)))

#my_model.add(Dropout(0.4))

my_model.add(Dense(1))

#def tanh(x):

#    return K.tanh(x)



#my_model.add(Activation(tanh))

#def softsign(x):

#    return x / (np.abs(x) + 1)



#my_model.add(Activation('softsign'))

def softplus(x):

    return np.log(np.exp(x) + 1)



my_model.add(Activation('softplus'))



my_model.compile(loss = 'mse',optimizer = 'adam', metrics = ['mean_squared_error'])

my_model.summary()
my_model.fit(X_train,y_train,batch_size = 4096,epochs = 10)
# creating submission file 

submission_pfs = my_model.predict(X_test)

# we will keep every value between 0 and 20

submission_pfs = submission_pfs.clip(0,20)

# creating dataframe with required columns 

submission = pd.DataFrame({'ID':test_data['ID'],'item_cnt_month':submission_pfs.ravel()})

# creating csv file from dataframe

submission.to_csv('sub_pfs.csv',index = False)