# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
sales_data = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')

item_cat = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')

items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')

shops = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')

sample_submission = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv')

test_data = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
# Displaying the sales data

sales_data.head()
# Displaying the test data

test_data.head()
def basic_eda(df):

    print("----------TOP 5 RECORDS--------")

    print(df.head(5))

    print("----------INFO-----------------")

    print(df.info())

    print("----------Describe-------------")

    print(df.describe())

    print("----------Columns--------------")

    print(df.columns)

    print("----------Data Types-----------")

    print(df.dtypes)

    print("-------Missing Values----------")

    print(df.isnull().sum())

    print("-------NULL values-------------")

    print(df.isna().sum())

    print("-----Shape Of Data-------------")

    print(df.shape)
#Litle bit of exploration of data



print("=============================Sales Data=============================")

basic_eda(sales_data)

print("=============================Test data=============================")

basic_eda(test_data)

print("=============================Item Categories=============================")

basic_eda(item_cat)

print("=============================Items=============================")

basic_eda(items)

print("=============================Shops=============================")

basic_eda(shops)

print("=============================Sample Submission=============================")

basic_eda(sample_submission)



sales_data['date'] = pd.to_datetime(sales_data['date'], format = '%d.%m.%Y')
sales_data.head()
dataset = sales_data.pivot_table(index=['shop_id', 'item_id'], values=['item_cnt_day'], columns=['date_block_num'], fill_value=0, aggfunc='sum')
dataset.head()
dataset.reset_index(inplace=True)
dataset.head()
dataset = pd.merge(test_data, dataset, on=['item_id', 'shop_id'], how='left')

dataset.head()
dataset.fillna(0,inplace=True)

dataset.head()
dataset.drop(['shop_id','item_id','ID'], inplace=True, axis=1)

dataset.head()
x_train = np.expand_dims(dataset.values[:,:-1], axis=2)

y_train = dataset.values[:,-1:]

x_test = np.expand_dims(dataset.values[:,1:], axis=2)

print(x_train.shape,y_train.shape,x_test.shape)
from keras import optimizers

from keras.utils import plot_model

from keras.models import Sequential, Model

from keras.layers.convolutional import Conv2D, MaxPooling1D, Conv1D

from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten, Dropout

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split
# Defining our model



model_lstm = Sequential()

model_lstm.add(LSTM(units=64, input_shape=(x_train.shape[1], x_train.shape[2])))

model_lstm.add(Dropout(0.4))

model_lstm.add(Dense(1))





model_lstm.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error'])

model_lstm.summary()
history_lstm = model_lstm.fit(x_train, y_train, batch_size=4096, epochs=50)
# Plot the loss curves for training

plt.plot(history_lstm.history['loss'], color='b', label="Training loss")

plt.legend(loc='best', shadow=True)
# creating submission file 

submission_pfs = model_lstm.predict(x_test)

# we will keep every value between 0 and 20

submission_pfs = submission_pfs.clip(0,20)

# creating dataframe with required columns 

submission = pd.DataFrame({'ID':test_data['ID'],'item_cnt_month':submission_pfs.ravel()})

# creating csv file from dataframe

submission.to_csv('sub_pfs.csv',index = False)
submission.head(3)
submission.shape, test_data.shape
#CNN for Time Series Forecasting



model_cnn = Sequential()

model_cnn.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2])))

model_cnn.add(MaxPooling1D(pool_size=2))

model_cnn.add(Flatten())

model_cnn.add(Dense(50, activation='relu'))

model_cnn.add(Dense(1))

model_cnn.compile(loss='mse', optimizer='adam')

model_cnn.summary()
cnn_history = model_cnn.fit(x_train, y_train, epochs=50, verbose=2)
plt.plot(cnn_history.history['loss'], color='b', label="Training loss")

plt.legend(loc='best', shadow=True)
#CNN-LSTM for Time Series Forecasting



#Reshape from [samples, timesteps, features] into [samples, subsequences, timesteps, features]



subsequences = 3

timesteps = x_train.shape[1]//subsequences

x_train_series_sub = x_train.reshape((x_train.shape[0], subsequences, timesteps, 1))

print('Train set shape', x_train_series_sub.shape)

model_cnn_lstm = Sequential()

model_cnn_lstm.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, x_train_series_sub.shape[2], x_train_series_sub.shape[3])))

model_cnn_lstm.add(TimeDistributed(MaxPooling1D(pool_size=2)))

model_cnn_lstm.add(TimeDistributed(Flatten()))

model_cnn_lstm.add(LSTM(50, activation='relu'))

model_cnn_lstm.add(Dense(1))

model_cnn_lstm.compile(loss='mse', optimizer='adam')

model_cnn_lstm.summary()
cnn_lstm_history = model_cnn_lstm.fit(x_train_series_sub, y_train, epochs=50, verbose=2)
plt.plot(cnn_lstm_history.history['loss'], color='b', label="Training loss")

plt.legend(loc='best', shadow=True)
#Comparing models



fig, axes = plt.subplots(2, 2, sharex=True, sharey=True,figsize=(22,12))

ax1, ax2 = axes[0]

ax3, ax4 = axes[1]





ax2.plot(cnn_history.history['loss'], label='Train loss')

ax2.legend(loc='best')

ax2.set_title('CNN')

ax2.set_xlabel('Epochs')

ax2.set_ylabel('MSE')



ax3.plot(history_lstm.history['loss'], label='Train loss')

ax3.legend(loc='best')

ax3.set_title('LSTM')

ax3.set_xlabel('Epochs')

ax3.set_ylabel('MSE')



ax4.plot(cnn_lstm_history.history['loss'], label='Train loss')

ax4.legend(loc='best')

ax4.set_title('CNN-LSTM')

ax4.set_xlabel('Epochs')

ax4.set_ylabel('MSE')



plt.show()