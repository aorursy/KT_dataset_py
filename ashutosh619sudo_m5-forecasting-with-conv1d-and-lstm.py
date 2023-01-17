# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv")
train.head()

train.shape
def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics: 

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max< np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
train = reduce_mem_usage(train)
calendar = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/calendar.csv")
calendar.head()
calendar = reduce_mem_usage(calendar)
import datetime as dt

date_index = calendar['date']

dates = date_index[0:1913]

dates_list = [dt.datetime.strptime(date, '%Y-%m-%d').date() for date in dates]
dates_list[:10]
train['item_store_id'] = train.apply(lambda x: x['item_id']+'_'+x['store_id'],axis=1)

DF_Sales = train.loc[:,'d_1':'d_1913'].T

DF_Sales.columns = train['item_store_id'].values

DF_Sales = pd.DataFrame(DF_Sales).set_index([dates_list])

DF_Sales.index = pd.to_datetime(DF_Sales.index)

DF_Sales.head()
for col in DF_Sales.columns[:5]:

    y = pd.DataFrame(DF_Sales.loc[:,col])

    y = pd.DataFrame(y).set_index([dates_list])

    

    y.index = pd.to_datetime(y.index)

    

    ax = y.plot(figsize=(30, 9),color='red')

    ax.set_facecolor('lightgrey')

    plt.xticks(fontsize=21 )

    plt.yticks(fontsize=21 )

    plt.legend(fontsize=20)

    plt.title(label = 'Sales Demand Selected Time Series Over Time',fontsize = 23)

    plt.ylabel(ylabel = 'Sales Demand',fontsize = 21)

    plt.xlabel(xlabel = 'Date',fontsize = 21)

    plt.show()

    
from sklearn.preprocessing import MinMaxScaler
data = np.array(y)

scaler = MinMaxScaler(feature_range=(0, 1))

dataset = scaler.fit_transform(data.reshape(-1, 1))
dataset
train_size = int(len(dataset) * 0.67)

test_size = len(dataset) - train_size

train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

train.shape,test.shape
def create_dataset(dataset, look_back=1):

    dataX, dataY = [], []

    for i in range(len(dataset)-look_back-1):

        a = dataset[i:(i+look_back), 0]

        dataX.append(a)

        dataY.append(dataset[i + look_back, 0])

    return np.array(dataX), np.array(dataY)
X_train,y_train = create_dataset(train,28)
X_test,y_test = create_dataset(test,28)
trainX = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

testX = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
trainX
import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import LSTM

from tensorflow.keras.layers import GRU

from tensorflow.keras.layers import Bidirectional

from tensorflow.keras.layers import Conv1D
model = Sequential()

model.add(Conv1D(filters=32, kernel_size=5,

                      strides=1, padding="causal",

                      activation="relu",

                      input_shape=[None, 1]))

model.add(LSTM(512))

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam',metrics=["mean_squared_error"])

model.fit(trainX, y_train, epochs=100,batch_size=1, verbose=2)
traning_pred = model.predict(trainX)
train_pred = pd.Series(scaler.inverse_transform(traning_pred).flatten())
plt.figure(num=None, figsize=(19, 6), facecolor='w', edgecolor='k')

plt.plot(train_pred)

plt.plot(train)

plt.legend(["Predicted","Real"])
test_pred = scaler.inverse_transform(model.predict(testX)).flatten()

plt.figure(num=None, figsize=(19, 6), dpi=80, facecolor='w', edgecolor='k')

plt.plot(test_pred)

plt.plot(test)

plt.legend(["Predicted","Real"])