# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

trainfile = os.path.join('/kaggle/input/demand-forecasting-kernels-only', 'train.csv')

testfile = os.path.join('/kaggle/input/demand-forecasting-kernels-only', 'test.csv')



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



train = pd.read_csv(trainfile, parse_dates=['date'], infer_datetime_format=True)

test = pd.read_csv(testfile, parse_dates=['date'], infer_datetime_format=True)



plt.plot(train['date'], train['sales'])



train['day'] = train['date'].map(lambda x: x.day)

train['month'] = train['date'].map(lambda x: x.month)

train['year'] = train['date'].map(lambda x: x.year)





test = pd.read_csv(testfile, parse_dates=['date'], infer_datetime_format=True)



test['day'] = test['date'].map(lambda x: x.day)

test['month'] = test['date'].map(lambda x: x.month)

test['year'] = test['date'].map(lambda x: x.year)





# In[13]:





train.drop(['date'], axis=1, inplace=True)

test.drop(['date'], axis=1, inplace=True)
train.shape, test.shape
# train.drop('year', axis=1, inplace=True)

# test.drop('year', axis=1, inplace=True)
train.sales.max()
train.head()
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()

train['sales'] = sc.fit_transform(train[['sales']])
train.columns



Y = train['sales'].values

X = train.drop('sales', axis=1)



X = np.asarray(X)

X = np.reshape(X, (X.shape[0], X.shape[1], 1))
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout

from keras.callbacks import EarlyStopping

from sklearn.model_selection import KFold



# 

# define 10-fold cross validation test harness

#kfold = KFold(n_splits=2, random_state=7)

cvscores = []

#for Xtrain, Xtest in kfold.split(X, Y):

regressor = Sequential()



# regressor.add(LSTM(units = 64, return_sequences=True, input_shape = (X.shape[1], 1)))

# regressor.add(LSTM(units = 32, return_sequences=True, input_shape = (X.shape[1], 1)))

regressor.add(LSTM(units = 16, return_sequences=True, input_shape = (X.shape[1], 1)))

regressor.add(LSTM(units = 8, input_shape = (X.shape[1], 1)))

regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))



regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])



es = EarlyStopping(monitor='val_loss', verbose=1, patience=10)

regressor.fit(X, Y, epochs = 20, batch_size = 10000, validation_split=0.2, callbacks=[es])



# evaluate the model

#scores = regressor.evaluate(X, Y, verbose=0)
test.drop('id', axis=1, inplace=True)

test.head()



X_test = np.asarray(test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
X_test.shape, X.shape
test_preds = regressor.predict(X_test)

testpreds = sc.inverse_transform(test_preds)



finalpreds = pd.DataFrame(testpreds, columns=['sales'])

finalpreds.index = np.arange(0, len(finalpreds))

finalpreds.index.name = 'id'
finalpreds.reset_index(inplace=True)

finalpreds.to_csv('submission.csv', index=False)