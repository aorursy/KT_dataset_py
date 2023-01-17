from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sys 

import numpy as np # linear algebra

from scipy.stats import randint

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL

import matplotlib.pyplot as plt # this is used for the plot the graph 

import seaborn as sns # used for plot interactive graph. 

from sklearn.model_selection import train_test_split # to split the data into two parts

from sklearn.model_selection import KFold # use for cross validation

from sklearn.preprocessing import StandardScaler # for normalization

from sklearn.preprocessing import MinMaxScaler

from sklearn.pipeline import Pipeline # pipeline making

from sklearn.model_selection import cross_val_score

from sklearn.feature_selection import SelectFromModel

from sklearn import metrics # for the check the error and accuracy of the model

from sklearn.metrics import mean_squared_error,r2_score



## for Deep-learing:

import keras

from keras.layers import Dense

from keras.models import Sequential

from keras.utils import to_categorical

from keras.optimizers import SGD 

from keras.callbacks import EarlyStopping

from keras.utils import np_utils

import itertools

from keras.layers import LSTM

from keras.layers.convolutional import Conv1D

from keras.layers.convolutional import MaxPooling1D

from keras.layers import Dropout
print(os.listdir('../input'))
df = pd.read_csv('../input/household_power_consumption.txt',sep = ';',

                parse_dates={'dt':['Date','Time']},

                infer_datetime_format=True,

                low_memory=False, na_values=['nan','?'],

                index_col='dt')
df.head()
droping_list_all=[]

for j in range(0, 7):

    if not df.iloc[:, j].notnull().all():

        droping_list_all.append(j)

droping_list_all
for j in range(0,7):

    df.iloc[:,j]=df.iloc[:,j].fillna(df.iloc[:,j].mean())
df.isnull().sum()
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):

    n_vars = 1 if type(data) is list else data.shape[1]

    dff = pd.DataFrame(data)

    cols, names = list(), list()

    for i in range(n_in, 0, -1):

        cols.append(dff.shift(-i))

        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

    for i in range(0, n_out):

        cols.append(dff.shift(-i))

        if i==0:

            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]

        else:

            names += [('var%d(t+%d)' % (j+1)) for j in range(n_vars)]

        

        agg = pd.concat(cols, axis=1)

        agg.columns = names

        if dropnan:

            agg.dropna(inplace=True)

        return agg
df_resample = df.resample('h').mean()

df_resample.shape
values = df_resample.values

scaler = MinMaxScaler(feature_range=(0, 1))

scaled = scaler.fit_transform(values)

reframed = series_to_supervised(scaled, 1, 1)

reframed.drop(reframed.columns[[8,9,10,11,12,13]], axis=1, inplace=True)

reframed.head()
values = reframed.values

n_train_time = 365*24

train = values[:n_train_time, :]

test = values[n_train_time:, :]

train_x, train_y = train[:, :-1], train[:, -1]

test_x, test_y = test[:, :-1], test[:, -1]

train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))

test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
model = Sequential()

model.add(LSTM(100, input_shape=(train_x.shape[1], train_x.shape[2])))

model.add(Dropout(0.2))

#    model.add(LSTM(70))

#    model.add(Dropout(0.3))

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')







# fit network

history = model.fit(train_x, train_y, epochs=20, batch_size=70, validation_data=(test_x, test_y), verbose=2, shuffle=False)



# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper right')

plt.show()



# make a prediction

yhat = model.predict(test_x)

test_x = test_x.reshape((test_x.shape[0], 7))

# invert scaling for forecast

inv_yhat = np.concatenate((yhat, test_x[:, -6:]), axis=1)

inv_yhat = scaler.inverse_transform(inv_yhat)

inv_yhat = inv_yhat[:,0]

# invert scaling for actual

test_y = test_y.reshape((len(test_y), 1))

inv_y = np.concatenate((test_y, test_x[:, -6:]), axis=1)

inv_y = scaler.inverse_transform(inv_y)

inv_y = inv_y[:,0]

# calculate RMSE

rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))

print('Test RMSE: %.3f' % rmse)
aa=[x for x in range(400)]

plt.plot(aa, inv_y[:400], marker='.', label="actual")

plt.plot(aa, inv_yhat[:400], 'r', label="prediction")

plt.ylabel('Global_active_power', size=15)

plt.xlabel('Time step', size=15)

plt.legend(fontsize=15)

plt.show()