# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

default_path = '../input/'
#checking files on input folder 
!ls ../input
train_df = pd.read_csv(default_path+'sales_train.csv')
items_df = pd.read_csv(default_path+'items.csv')
test_df = pd.read_csv(default_path+'test.csv')
#We dont need other files, you can use them in order to do more EDA
# id seems to be usless
item_categories_df = pd.read_csv(default_path+'item_categories.csv')
shops_df = pd.read_csv(default_path+'shops.csv')
print(train_df.shape, test_df.shape)
train_df.head()
#Adding date to the training set
train_df['date'] = pd.to_datetime(train_df['date'], format='%d.%m.%Y')
#data in test is not in training set , the missing data will be filled with zeros
dataset = train_df.pivot_table(index=['item_id', 'shop_id'],values=['item_cnt_day'], columns='date_block_num', fill_value=0)
dataset.shape
dataset = dataset.reset_index()
dataset.head()
#Lets add items id and shop id to the test dataset
dataset = pd.merge(test_df, dataset, on=['item_id', 'shop_id'], how='left')
dataset = dataset.fillna(0)
dataset.head()
X_train = np.expand_dims(dataset.values[:, :-1], axis=2)
y_train = dataset.values[:, -1:]

X_test = np.expand_dims(dataset.values[:, 1:], axis=2)
print(X_train.shape, y_train.shape, X_test.shape)
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
model = Sequential()
model.add(LSTM(units=64, input_shape=(36, 1)))
model.add(Dropout(0.3))
model.add(Dense(1))

model.compile(loss='mse',
              optimizer='Adagrad',
              metrics=['mean_squared_error'])
model.summary() 
history = model.fit(X_train, y_train, batch_size=4096, epochs=10)
plt.plot(history.history['loss'], label= 'loss(mse)')
plt.plot(np.sqrt(history.history['mean_squared_error']), label= 'rmse')
plt.legend(loc=1)
LSTM_prediction = model.predict(X_test)
submission = pd.DataFrame({'ID': test_df['ID'], 'item_cnt_month': LSTM_prediction.ravel()})
submission.to_csv('submission.csv',index=False)
