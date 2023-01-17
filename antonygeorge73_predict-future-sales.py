#importing libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#Reading the data

sales_data = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')
item_cat = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')
items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')
shops = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')
sample_submission = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv')
test_data = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
print(" ***********Sales Data************")
print(sales_data.head(5))

print("\n***********Item Categories************")
print(item_cat.head(5))

print("\n**********************Items***********************")
print(items.head(5))

print("\n***********Shops*********************")
print(shops.head(5))

print("\n***********Test Data************")
print(test_data.head(5))
print(" ***********Sales Data************")
print(sales_data.info())

print("\n***********Item Categories************")
print(item_cat.info())

print("\n**************Items***********************")
print(items.info())

print("\n***********Shops************")
print(shops.info())

print("\n***********Test Data************")
print(test_data.info())


#Converting object to date

import datetime
sales_data['date'] = pd.to_datetime(sales_data['date'],format = '%d.%m.%Y')
sales_data.head(5)
dataset = sales_data.pivot_table(index = {'shop_id','item_id'},values = {'item_cnt_day'},columns = {'date_block_num'},fill_value = 0, aggfunc= 'sum')
dataset.head(5)
#resetting the indices
dataset.reset_index(inplace = True)
dataset.head(5)
#Merging pivot table with the test data

dataset = pd.merge(test_data, dataset,on = ['item_id','shop_id'],how='left')
dataset.head(5)
dataset.fillna(0, inplace=True)
dataset.head(5)
#Droping columns

dataset.drop({'shop_id','item_id','ID'}, inplace=True, axis=1)
dataset.head(5)
X_train = np.expand_dims(dataset.values[:,:-1],axis=2)

y_train = dataset.values[:,-1:]

X_test = np.expand_dims(dataset.values[:,1:],axis=2)

print(X_train.shape, y_train.shape, X_test.shape)
from keras import optimizers
from keras.utils import plot_model
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten,  Dropout
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
#Defining the model

model_lstm = Sequential()
model_lstm.add(LSTM(units = 64,input_shape = (X_train.shape[1], X_train.shape[2])))
model_lstm.add(Dropout(0.4))
model_lstm.add(Dense(1))

model_lstm.compile(loss = 'mse', optimizer = 'adam', metrics = ['mean_squared_error'])
model_lstm.summary()
history_lstm = model_lstm.fit(X_train, y_train, batch_size = 4096, epochs = 10)
#Plotting the loss
import matplotlib.pyplot as plt
plt.plot(history_lstm.history['loss'], color='b', label = "Training loss")
plt.legend(loc='best')
#Creating submission file
submission_pfs = model_lstm.predict(X_test)

submission_pfs = submission_pfs.clip(0,20)

submission = pd.DataFrame({'ID':test_data['ID'], 'item_cnt_month':submission_pfs.ravel()})

submission.to_csv('sub_pfs.csv', index=False)

submission.head(5)
submission.shape, test_data.shape


