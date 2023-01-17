# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score as R2_score

from sklearn.preprocessing import MinMaxScaler, Normalizer
bs = 32

ts = 8

n_features = 3

epochs = 25
data_train = pd.read_csv('../input/daily-climate-time-series-data/DailyDelhiClimateTrain.csv',sep=',')

data_test = pd.read_csv('../input/daily-climate-time-series-data/DailyDelhiClimateTest.csv',sep=',')

data = pd.concat([data_train,data_test])



dates_test = data_test.index.to_numpy()[ts-1:]



data.info()
data['date'] = pd.to_datetime(data['date'])

data = data.rename(columns={"meantemp":"temp","wind_speed":"wind","meanpressure":"pressure"})

data.head()
eps = 1e-11



NAN = np.NAN



# Logarithmic transformation



def transform_logratios(serie):

    aux = np.log((serie[1:]+eps) / (serie[0:-1]+eps))

    return np.hstack( ([NAN], aux))

def inverse_transform_logratios(log_ratio, temp_prev):

    return np.multiply(temp_prev, np.exp(log_ratio))



transform = transform_logratios

inverse_transform = inverse_transform_logratios

scaler = MinMaxScaler()
transformed_X = scaler.fit_transform(data.loc[:, ["humidity","wind","pressure"]])

#transformed_X = pd.DataFrame(transformed_X, columns = ["humidity_t","wind_t","pressure_t"])

transformed_X.shape
transformed_y = np.log(data.temp).diff().to_numpy().reshape((data.temp.shape[0],1))

transformed_y.shape
def winnowing_X(series, n_features, ts):

    n_samples = series.shape[0]-ts+1

    X_data = np.empty((n_samples, ts, n_features))

    for i, e in enumerate(series):

        if i >= n_samples: break

        for j in range(ts):

            X_data[i][j] = e.copy()

    return X_data



def winnowing_y(series, n_features, ts):

    n_samples = series.shape[0]-ts+1

    y_data = np.empty((n_samples, 1, 1))

    for i, e in enumerate(series):

        if i < ts: continue

        if i-ts >= n_samples: break

        y_data[i-ts][0] = e.copy()

    return y_data
X_train = winnowing_X(transformed_X[:data_train.shape[0]], transformed_X.shape[1], ts)

y_train = winnowing_y(transformed_y[:data_train.shape[0]], 1, ts)

X_test = winnowing_X(transformed_X[data_train.shape[0]:], transformed_X.shape[1], ts)

y_test = winnowing_y(transformed_y[data_train.shape[0]:], 1, ts)
from keras.models import Sequential, load_model

from keras.layers import Dense, LSTM, GRU

from keras.callbacks import ModelCheckpoint, TensorBoard

from keras.optimizers import SGD



from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score as R2_score

from sklearn.preprocessing import MinMaxScaler
model = Sequential()

model.add(GRU(16, input_shape=(ts, n_features),

#              kernel_regularizer='l1'

              )

         )

model.add(Dense(1,

#                kernel_regularizer='l1'

               )

         )

opt = SGD(learning_rate=0.01, momentum=0.1)

model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse']) # 'RMSprop'



model.summary()
history = model.fit(X_train, y_train,validation_split=0.33, epochs=epochs, batch_size=bs, verbose=1)
print(history.history.keys())



import matplotlib.pyplot as plt

#summary mse

plt.plot(history.history['mse'])

plt.plot(history.history['val_mse'])

plt.title('model mse')

plt.ylabel('mse')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

#summary loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
y_test_pred = model.predict(X_test)
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator



from IPython.display import display_html
temp_prev_test = data_test["meantemp"][ts:]



plt.figure(figsize=(15,5))

plt.plot(dates_test[:-1], inverse_transform(y_test_pred.flatten()[:-1],temp_prev_test), '--', c='royalblue',

         label='pred')

plt.plot(dates_test[:-1], inverse_transform(y_test.flatten()[:-1],temp_prev_test), '--', c='orange',

         label='true')