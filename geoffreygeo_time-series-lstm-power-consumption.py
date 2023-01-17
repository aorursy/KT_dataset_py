# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



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



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/household_power_consumption.txt',sep=';')
display(df.head())

display(df.isnull().sum())
#merging two calloumns data and time 

#missing values 

df = pd.read_csv('../input/household_power_consumption.txt', sep=';', 

                 parse_dates={'dt' : ['Date', 'Time']}, infer_datetime_format=True, 

                 low_memory=False, na_values=['nan','?'], index_col='dt')
df.head()
display(df.dtypes)

display(df.info())

display(df.describe())
droping_list_all=[]

for j in range(0,7):

    if not df.iloc[:, j].notnull().all():

        droping_list_all.append(j)        

        #print(df.iloc[:,j].unique())

droping_list_all


for j in range(0,7):        

        df.iloc[:,j]=df.iloc[:,j].fillna(df.iloc[:,j].mean())
df.isnull().sum()
df.Global_active_power.resample('D').sum().plot(title='Global_active_power resampled over day for sum') 



plt.tight_layout()

plt.show()   



df.Global_active_power.resample('D').mean().plot(title='Global_active_power resampled over day for mean', color='red') 

plt.tight_layout()

plt.show()
r = df.Global_intensity.resample('D').agg(['mean', 'std'])

r.plot(subplots = True, title='Global_intensity resampled over day')

plt.show()
r2 = df.Global_reactive_power.resample('D').agg(['mean', 'std'])

r2.plot(subplots = True, title='Global_reactive_power resampled over day', color='red')

plt.show()
# Correlations among columns

plt.matshow(df.corr(method='spearman'),vmax=1,vmin=-1,cmap='PRGn')

plt.title('without resampling', size=15)

plt.colorbar()

plt.show()
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):

	n_vars = 1 if type(data) is list else data.shape[1]

	dff = pd.DataFrame(data)

	cols, names = list(), list()

	# input sequence (t-n, ... t-1)

	for i in range(n_in, 0, -1):

		cols.append(dff.shift(i))

		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

	# forecast sequence (t, t+1, ... t+n)

	for i in range(0, n_out):

		cols.append(dff.shift(-i))

		if i == 0:

			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]

		else:

			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

	# put it all together

	agg = pd.concat(cols, axis=1)

	agg.columns = names

	# drop rows with NaN values

	if dropnan:

		agg.dropna(inplace=True)

	return agg

 
df_resample = df.resample('h').mean() 

df_resample.shape

values = df_resample.values 





## full data without resampling

#values = df.values



# integer encode direction

# ensure all data is float

#values = values.astype('float32')

# normalize features

scaler = MinMaxScaler(feature_range=(0, 1))

scaled = scaler.fit_transform(values)

# frame as supervised learning

reframed = series_to_supervised(scaled, 1, 1)



# drop columns we don't want to predict

reframed.drop(reframed.columns[[8,9,10,11,12,13]], axis=1, inplace=True)

print(reframed.head())
# split into train and test sets

values = reframed.values



n_train_time = 365*24

train = values[:n_train_time, :]

test = values[n_train_time:, :]

##test = values[n_train_time:n_test_time, :]

# split into input and outputs

train_X, train_y = train[:, :-1], train[:, -1]

test_X, test_y = test[:, :-1], test[:, -1]

# reshape input to be 3D [samples, timesteps, features]

train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))

test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape) 

# We reshaped the input into the 3D format as expected by LSTMs, namely [samples, timesteps, features].
model = Sequential()

model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))

model.add(Dropout(0.2))



model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.summary()
# fit network

history = model.fit(train_X, train_y, epochs=20, batch_size=70, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper right')

plt.show()



# make a prediction

yhat = model.predict(test_X)

test_X = test_X.reshape((test_X.shape[0], 7))

# invert scaling for forecast

inv_yhat = np.concatenate((yhat, test_X[:, -6:]), axis=1)

inv_yhat = scaler.inverse_transform(inv_yhat)

inv_yhat = inv_yhat[:,0]

# invert scaling for actual

test_y = test_y.reshape((len(test_y), 1))

inv_y = np.concatenate((test_y, test_X[:, -6:]), axis=1)

inv_y = scaler.inverse_transform(inv_y)

inv_y = inv_y[:,0]

# calculate RMSE

rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))

print('Test RMSE: %.3f' % rmse)
aa=[x for x in range(200)]

plt.plot(aa, inv_y[:200], marker='.', label="actual")

plt.plot(aa, inv_yhat[:200], 'r', label="prediction")

plt.ylabel('Global_active_power', size=15)

plt.xlabel('Time step', size=15)

plt.legend(fontsize=15)

plt.show()