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

import matplotlib.pyplot as plt

%matplotlib inline

from statsmodels.tools.eval_measures import rmse

from sklearn.preprocessing import MinMaxScaler

from keras.preprocessing.sequence import TimeseriesGenerator

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout

import warnings

warnings.filterwarnings("ignore")
df=pd.read_csv("../input/covid19/covid_19_data.csv")

df.head()
df['ObservationDate'] = pd.to_datetime(df["ObservationDate"])

df_idx = df.set_index(["ObservationDate"], drop=True)

df_idx.head(5)
df_idx = df_idx.sort_index(axis=1, ascending=True)

df_idx = df_idx.iloc[::-1]
data = df_idx[['Deaths']]

data.plot(y='Deaths')
split_date = pd.Timestamp('22-01-2020')



train = data.loc[:split_date]

test = data.loc[split_date:]



ax = train.plot(figsize=(10,12))

test.plot(ax=ax)

plt.legend(['train', 'test'])

plt.show()


from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()

train_sc = sc.fit_transform(train)

test_sc = sc.transform(test)


train_sc_df = pd.DataFrame(train_sc, columns=['Y'], index=train.index)

test_sc_df = pd.DataFrame(test_sc, columns=['Y'], index=test.index)



for s in range(1,2):

    train_sc_df['X_{}'.format(s)] = train_sc_df['Y'].shift(s)

    test_sc_df['X_{}'.format(s)] = test_sc_df['Y'].shift(s)



X_train = train_sc_df.dropna().drop('Y', axis=1)

y_train = train_sc_df.dropna().drop('X_1', axis=1)



X_test = test_sc_df.dropna().drop('Y', axis=1)

y_test = test_sc_df.dropna().drop('X_1', axis=1)



X_train = X_train.as_matrix()

y_train = y_train.as_matrix()



X_test = X_test.as_matrix()

y_test = y_test.as_matrix()


print('Train size: (%d x %d)'%(X_train.shape[0], X_train.shape[1]))

print('Test size: (%d x %d)'%(X_test.shape[0], X_test.shape[1]))


from sklearn.svm import SVR

regressor = SVR(kernel='rbf')
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
plt.plot(y_test)

plt.plot(y_pred)

plt.legend(['y_test','y_pred'])
from sklearn.metrics import r2_score



def adj_r2_score(r2, n, k):

    return 1-((1-r2)*((n-1)/(n-k-1)))



r2_test = r2_score(y_test, y_pred)

print("R-squared is: %f"%r2_test)


from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import Adam

import keras.backend as K


K.clear_session()

model = Sequential()

model.add(Dense(1, input_shape=(X_test.shape[1],), activation='tanh', kernel_initializer='lecun_uniform'))

model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error')

model.fit(X_train, y_train, batch_size=16, epochs=20, verbose=1)
y_pred = model.predict(X_test)

plt.plot(y_test)

plt.plot(y_pred)

plt.legend(['y_test','y_pred'])

print('R-Squared: %f'%(r2_score(y_test, y_pred)))


K.clear_session()

model = Sequential()

model.add(Dense(50, input_shape=(X_test.shape[1],), activation='relu', kernel_initializer='lecun_uniform'))

model.add(Dense(50, input_shape=(X_test.shape[1],), activation='relu'))

model.add(Dense(1))

model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error')

model.fit(X_train, y_train, batch_size=16, epochs=20, verbose=1)
y_pred = model.predict(X_test)

plt.plot(y_test)

plt.plot(y_pred)

plt.legend(['y_test','y_pred'])

print('R-Squared: %f'%(r2_score(y_test, y_pred)))