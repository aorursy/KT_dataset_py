import warnings

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from keras import optimizers

from keras.utils import plot_model

from keras.models import Sequential, Model

from keras.layers.convolutional import Conv1D, MaxPooling1D

from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

import plotly.plotly as py

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot



%matplotlib inline

warnings.filterwarnings("ignore")

init_notebook_mode(connected=True)



# Set seeds to make the experiment more reproducible.

from tensorflow import set_random_seed

from numpy.random import seed

set_random_seed(1)

seed(1)
train = pd.read_csv('../input/demand-forecasting-kernels-only/train.csv', parse_dates=['date'])

test = pd.read_csv('../input/demand-forecasting-kernels-only/test.csv', parse_dates=['date'])
train.describe()
train.head()
print('Min date from train set: %s' % train['date'].min().date())

print('Max date from train set: %s' % train['date'].max().date())
lag_size = (test['date'].max().date() - train['date'].max().date()).days

print('Max date from train set: %s' % train['date'].max().date())

print('Max date from test set: %s' % test['date'].max().date())

print('Forecast lag size', lag_size)
daily_sales = train.groupby('date', as_index=False)['sales'].sum()

store_daily_sales = train.groupby(['store', 'date'], as_index=False)['sales'].sum()

item_daily_sales = train.groupby(['item', 'date'], as_index=False)['sales'].sum()
daily_sales_sc = go.Scatter(x=daily_sales['date'], y=daily_sales['sales'])

layout = go.Layout(title='Daily sales', xaxis=dict(title='Date'), yaxis=dict(title='Sales'))

fig = go.Figure(data=[daily_sales_sc], layout=layout)

iplot(fig)
store_daily_sales_sc = []

for store in store_daily_sales['store'].unique():

    current_store_daily_sales = store_daily_sales[(store_daily_sales['store'] == store)]

    store_daily_sales_sc.append(go.Scatter(x=current_store_daily_sales['date'], y=current_store_daily_sales['sales'], name=('Store %s' % store)))



layout = go.Layout(title='Store daily sales', xaxis=dict(title='Date'), yaxis=dict(title='Sales'))

fig = go.Figure(data=store_daily_sales_sc, layout=layout)

iplot(fig)
item_daily_sales_sc = []

for item in item_daily_sales['item'].unique():

    current_item_daily_sales = item_daily_sales[(item_daily_sales['item'] == item)]

    item_daily_sales_sc.append(go.Scatter(x=current_item_daily_sales['date'], y=current_item_daily_sales['sales'], name=('Item %s' % item)))



layout = go.Layout(title='Item daily sales', xaxis=dict(title='Date'), yaxis=dict(title='Sales'))

fig = go.Figure(data=item_daily_sales_sc, layout=layout)

iplot(fig)
train = train[(train['date'] >= '2017-01-01')]
train_gp = train.sort_values('date').groupby(['item', 'store', 'date'], as_index=False)

train_gp = train_gp.agg({'sales':['mean']})

train_gp.columns = ['item', 'store', 'date', 'sales']

train_gp.head()
def series_to_supervised(data, window=1, lag=1, dropnan=True):

    cols, names = list(), list()

    # Input sequence (t-n, ... t-1)

    for i in range(window, 0, -1):

        cols.append(data.shift(i))

        names += [('%s(t-%d)' % (col, i)) for col in data.columns]

    # Current timestep (t=0)

    cols.append(data)

    names += [('%s(t)' % (col)) for col in data.columns]

    # Target timestep (t=lag)

    cols.append(data.shift(-lag))

    names += [('%s(t+%d)' % (col, lag)) for col in data.columns]

    # Put it all together

    agg = pd.concat(cols, axis=1)

    agg.columns = names

    # Drop rows with NaN values

    if dropnan:

        agg.dropna(inplace=True)

    return agg
window = 29

lag = lag_size

series = series_to_supervised(train_gp.drop('date', axis=1), window=window, lag=lag)

series.head()
last_item = 'item(t-%d)' % window

last_store = 'store(t-%d)' % window

series = series[(series['store(t)'] == series[last_store])]

series = series[(series['item(t)'] == series[last_item])]
columns_to_drop = [('%s(t+%d)' % (col, lag)) for col in ['item', 'store']]

for i in range(window, 0, -1):

    columns_to_drop += [('%s(t-%d)' % (col, i)) for col in ['item', 'store']]

series.drop(columns_to_drop, axis=1, inplace=True)

series.drop(['item(t)', 'store(t)'], axis=1, inplace=True)
# Label

labels_col = 'sales(t+%d)' % lag_size

labels = series[labels_col]

series = series.drop(labels_col, axis=1)



X_train, X_valid, Y_train, Y_valid = train_test_split(series, labels.values, test_size=0.4, random_state=0)

print('Train set shape', X_train.shape)

print('Validation set shape', X_valid.shape)

X_train.head()
epochs = 40

batch = 256

lr = 0.0003

adam = optimizers.Adam(lr)
model_mlp = Sequential()

model_mlp.add(Dense(100, activation='relu', input_dim=X_train.shape[1]))

model_mlp.add(Dense(1))

model_mlp.compile(loss='mse', optimizer=adam)

model_mlp.summary()
mlp_history = model_mlp.fit(X_train.values, Y_train, validation_data=(X_valid.values, Y_valid), epochs=epochs, verbose=2)
X_train_series = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))

X_valid_series = X_valid.values.reshape((X_valid.shape[0], X_valid.shape[1], 1))

print('Train set shape', X_train_series.shape)

print('Validation set shape', X_valid_series.shape)
model_cnn = Sequential()

model_cnn.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train_series.shape[1], X_train_series.shape[2])))

model_cnn.add(MaxPooling1D(pool_size=2))

model_cnn.add(Flatten())

model_cnn.add(Dense(50, activation='relu'))

model_cnn.add(Dense(1))

model_cnn.compile(loss='mse', optimizer=adam)

model_cnn.summary()
cnn_history = model_cnn.fit(X_train_series, Y_train, validation_data=(X_valid_series, Y_valid), epochs=epochs, verbose=2)
model_lstm = Sequential()

model_lstm.add(LSTM(50, activation='relu', input_shape=(X_train_series.shape[1], X_train_series.shape[2])))

model_lstm.add(Dense(1))

model_lstm.compile(loss='mse', optimizer=adam)

model_lstm.summary()
lstm_history = model_lstm.fit(X_train_series, Y_train, validation_data=(X_valid_series, Y_valid), epochs=epochs, verbose=2)
subsequences = 2

timesteps = X_train_series.shape[1]//subsequences

X_train_series_sub = X_train_series.reshape((X_train_series.shape[0], subsequences, timesteps, 1))

X_valid_series_sub = X_valid_series.reshape((X_valid_series.shape[0], subsequences, timesteps, 1))

print('Train set shape', X_train_series_sub.shape)

print('Validation set shape', X_valid_series_sub.shape)
model_cnn_lstm = Sequential()

model_cnn_lstm.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, X_train_series_sub.shape[2], X_train_series_sub.shape[3])))

model_cnn_lstm.add(TimeDistributed(MaxPooling1D(pool_size=2)))

model_cnn_lstm.add(TimeDistributed(Flatten()))

model_cnn_lstm.add(LSTM(50, activation='relu'))

model_cnn_lstm.add(Dense(1))

model_cnn_lstm.compile(loss='mse', optimizer=adam)
cnn_lstm_history = model_cnn_lstm.fit(X_train_series_sub, Y_train, validation_data=(X_valid_series_sub, Y_valid), epochs=epochs, verbose=2)
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True,figsize=(22,12))

ax1, ax2 = axes[0]

ax3, ax4 = axes[1]



ax1.plot(mlp_history.history['loss'], label='Train loss')

ax1.plot(mlp_history.history['val_loss'], label='Validation loss')

ax1.legend(loc='best')

ax1.set_title('MLP')

ax1.set_xlabel('Epochs')

ax1.set_ylabel('MSE')



ax2.plot(cnn_history.history['loss'], label='Train loss')

ax2.plot(cnn_history.history['val_loss'], label='Validation loss')

ax2.legend(loc='best')

ax2.set_title('CNN')

ax2.set_xlabel('Epochs')

ax2.set_ylabel('MSE')



ax3.plot(lstm_history.history['loss'], label='Train loss')

ax3.plot(lstm_history.history['val_loss'], label='Validation loss')

ax3.legend(loc='best')

ax3.set_title('LSTM')

ax3.set_xlabel('Epochs')

ax3.set_ylabel('MSE')



ax4.plot(cnn_lstm_history.history['loss'], label='Train loss')

ax4.plot(cnn_lstm_history.history['val_loss'], label='Validation loss')

ax4.legend(loc='best')

ax4.set_title('CNN-LSTM')

ax4.set_xlabel('Epochs')

ax4.set_ylabel('MSE')



plt.show()
mlp_train_pred = model_mlp.predict(X_train.values)

mlp_valid_pred = model_mlp.predict(X_valid.values)

print('Train rmse:', np.sqrt(mean_squared_error(Y_train, mlp_train_pred)))

print('Validation rmse:', np.sqrt(mean_squared_error(Y_valid, mlp_valid_pred)))
cnn_train_pred = model_cnn.predict(X_train_series)

cnn_valid_pred = model_cnn.predict(X_valid_series)

print('Train rmse:', np.sqrt(mean_squared_error(Y_train, cnn_train_pred)))

print('Validation rmse:', np.sqrt(mean_squared_error(Y_valid, cnn_valid_pred)))
lstm_train_pred = model_lstm.predict(X_train_series)

lstm_valid_pred = model_cnn.predict(X_valid_series)

print('Train rmse:', np.sqrt(mean_squared_error(Y_train, lstm_train_pred)))

print('Validation rmse:', np.sqrt(mean_squared_error(Y_valid, lstm_valid_pred)))
cnn_lstm_train_pred = model_cnn_lstm.predict(X_train_series_sub)

cnn_lstm_valid_pred = model_cnn_lstm.predict(X_valid_series_sub)

print('Train rmse:', np.sqrt(mean_squared_error(Y_train, cnn_lstm_train_pred)))

print('Validation rmse:', np.sqrt(mean_squared_error(Y_valid, cnn_lstm_valid_pred)))