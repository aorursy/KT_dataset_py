# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

import math

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



filepath = None

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        filepath = os.path.join(dirname, filename)



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataframe = pd.read_csv(filepath)

dataset_ = dataframe.values

# dataset = dataset.astype('float32')
num_train = 97

num_test = 48

train = np.zeros((97, 1))

test = np.zeros((48, 1))

dataset = np.zeros((145, 1))

for i, value in enumerate(dataset_):

    if i < 97:

        train[i] = value[1]

    elif i < 145:

        test[i-97] = value[1]

    dataset[i] = value[1]
def shift(arr, num, fill_value=0):

    #print(arr)

    result = np.empty_like(arr)

    if num > 0:

        result[:num] = fill_value

        result[num:] = arr[:-num]

    elif num < 0:

        result[num:] = fill_value

        result[:num] = arr[-num:]

    else:

        result = arr

    return result



def mvts_to_xy(*tslist, n_x=1, n_y=1, x_idx=None, y_idx=None):

    n_ts = len(tslist)

    if n_ts == 0:

        raise ValueError('At least one timeseries required as input')



    #TODO: Validation of other options



    result = []



    for ts in tslist:

        ts_cols = 1 if ts.ndim==1 else ts.shape[1]

        if x_idx is None:

            x_idx = range(0,ts_cols)

        if y_idx is None:

            y_idx = range(0,ts_cols)



        n_x_vars = len(x_idx)

        n_y_vars = len(y_idx)



        ts_rows = ts.shape[0]

        n_rows = ts_rows - n_x - n_y + 1



        dataX=np.empty(shape=(n_rows, n_x_vars * n_x),dtype=np.float32)

        dataY=np.empty(shape=(n_rows, n_y_vars * n_y),dtype=np.float32)

        x_cols, y_cols, names = list(), list(), list()



        # input sequence x (t-n, ... t-1)

        from_col = 0

        for i in range(n_x, 0, -1):

            dataX[:,from_col:from_col+n_x_vars]=shift(ts[:,x_idx],i)[n_x:ts_rows-n_y+1]

            from_col = from_col+n_x_vars



        # forecast sequence (t, t+1, ... t+n)

        from_col = 0

        for i in range(0, n_y):

            #y_cols.append(shift(ts,-i))

            dataY[:,from_col:from_col+n_y_vars]=shift(ts[:,y_idx],-i)[n_x:ts_rows-n_y+1]

            from_col = from_col + n_y_vars



        # put it all together

        #x_agg = concat(x_cols, axis=1).dropna(inplace=True)

        #y_agg = concat(y_cols, axis=1).dropna(inplace=True)



        #dataX = np.array(x_cols,dtype=np.float32)

        #dataY = np.array(y_cols,dtype=np.float32)



        result.append(dataX)

        result.append(dataY)

    return result



# reshape into X=t-1,t and Y=t+1

n_x=2

n_y=1



X_train, Y_train, X_test, Y_test = mvts_to_xy(train,test,n_x=n_x,n_y=n_y)
import tensorflow as tf



import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import SGD
keras.backend.clear_session()
num_layers = 2

num_neurons = [8,8]

n_epochs = 50

batch_size = 2



model = Sequential()

model.add(Dense(num_neurons[0], activation='relu', input_shape=(n_x,)))

model.add(Dense(num_neurons[1], activation='relu'))

model.add(Dense(units=1))

model.summary()



model.compile(loss='mse', optimizer='adam')



model.fit(X_train, Y_train,

                    batch_size=batch_size,

                    epochs=n_epochs)



score = model.evaluate(X_test, Y_test)

print('\nTest mse:', score)

print('Test rmse:', math.sqrt(score))
import matplotlib.pyplot as plt



# make predictions

Y_train_pred = model.predict(X_train)

Y_test_pred = model.predict(X_test)



# shift train predictions for plotting

Y_train_pred_plot = np.empty_like(dataset)

Y_train_pred_plot[:, :] = np.nan

Y_train_pred_plot[n_x-1:len(Y_train_pred)+n_x-1, :] = Y_train_pred



# shift test predictions for plotting

Y_test_pred_plot = np.empty_like(dataset)

Y_test_pred_plot[:, :] = np.nan

Y_test_pred_plot[len(Y_train_pred)+(n_x*2)-1:len(dataset)-1, :] = Y_test_pred



# plot baseline and predictions

plt.plot(dataset,label='Original Data')

plt.plot(Y_train_pred_plot,label='Y_train_pred')

plt.plot(Y_test_pred_plot,label='Y_test_pred')

plt.legend()

plt.show()