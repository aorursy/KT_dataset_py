# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # Graph plot

import seaborn as sns # Graph plot

import warnings # Ignore warnings

warnings.filterwarnings("ignore")

from keras.models import Sequential

# Libraries for RNN (LSTM)

from keras.layers import LSTM

from keras.layers import Dense

from keras.layers import Dropout



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Importing datasets

item_cat = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv")

items = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/items.csv")

sales_train = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv")

shops = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/shops.csv")

test = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/test.csv")
item_cat.head()
item_cat.info()
items.head()
items.info()
shops.head()
shops.info()
sales_train.head()  #Top5 details of training dataset
sales_train.info()   #Information about training dataset
sales_train.shape  # No. of rows & columns in dataset
sales_train.isnull().sum() #Checking missing values
#Coverting Date from object type to Datetime format

sales_train['date'] = pd.to_datetime(sales_train['date'],format = '%d.%m.%Y')

sales_train.info()  #Checking information of dataset again
sales_train.head()
test.head()   #Top5 details of test dataset
test.info()  #Information about test dataset
test.shape  #No. of rows & columns in test dataset
test.isnull().sum()   #Checking missing values in test dataset
# Pivot table to get monthly sales count

sales_monthly = sales_train.pivot_table(index=['shop_id', 'item_id'], values=['item_cnt_day'], columns=['date_block_num'], aggfunc='sum').reset_index()
sales_monthly.fillna(0, inplace=True) #Replacing null values by zero

sales_monthly.head()
#Combining Training and Test dataset

df_combined = pd.merge(test, sales_monthly, how='left', on=['item_id', 'shop_id'])

df_combined.fillna(0, inplace=True) #Replacing null values by zero

df_combined.head()  #Printing top5 details of combined dataset
#Dropping shop_id and item_id as these are not required in output

df_combined.drop(['shop_id', 'item_id'], inplace=True, axis=1)
#Defining Training and Test for model preparation

X_train = df_combined.values[:, 1:-1]  #All columns except ID and last column i.e, items sold in months excluding last month

y_train = df_combined.values[:, -1:]   #Last column i.e, items sold in last month

X_test = df_combined.values[:, 2:]     #All columns except ID and (item_cnt_day,0) i.e, items sold in months exclusing first month
#Reshaping X_train and X_test to 3-dimensional as it is required for LSTM (RNN)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

print(X_train.shape, X_test.shape)
#Preparing model

model = Sequential()



#Adding first LSTM layer

model.add(LSTM(units=50, return_sequences=False, input_shape=(33,1)))

model.add(Dropout(0.2))



'''#Adding second LSTM layer

model.add(LSTM(units=50, return_sequences=True)

model.add(Dropout(0.2))



#Adding third LSTM layer

model.add(LSTM(units=50, return_sequences=True)

model.add(Dropout(0.2))



#Adding fourth LSTM layer

model.add(LSTM(units=50, return_sequences=False)

model.add(Dropout(0.2))'''

         

#Adding the output layer

model.add(Dense(units=1))



#Compiling the LSTM (RNN)

model.compile(optimizer='adam', loss='mean_squared_error')

          

model.summary()   
model.fit(X_train, y_train, epochs=15, batch_size=32)  #Fitting model on training dataset
result = model.predict(X_test)  #Predicting result
#Creating submission file

submission = pd.DataFrame({'ID':test['ID'], 'item_cnt_month':result.ravel()})  # ravel(): converting to 1D array 

                           

#Creating csv file

submission.to_csv('submission_future_sales_predict.csv', index=False)  