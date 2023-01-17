!pip install chart_studio
import numpy as np

import pandas as pd

import os

import matplotlib.pylab as plt

import chart_studio.plotly as py

import plotly.graph_objs as go

from plotly.offline import iplot



from sklearn import model_selection

import tensorflow as tf

from keras import optimizers, regularizers

from keras.models import Model

from keras.layers import Dense, Input, LSTM, Conv1D, Activation, MaxPooling1D, Flatten

import keras.backend as K



%matplotlib inline
bpath = '../input/competitive-data-science-predict-future-sales/'

train = pd.read_csv(bpath + 'sales_train.csv', parse_dates=['date'], infer_datetime_format=True)

train.head()
train.info()
train['sales'] = train.item_price * train.item_cnt_day

train.head()
daily_sales = train.groupby('date', as_index=False)['sales'].sum()

daily_sales = daily_sales.sort_values('date', axis=0)

daily_sales.head()
daily_sales_sp = go.Scatter(x=daily_sales.date, y=daily_sales.sales)

layout = go.Layout(title='Daily Sales', xaxis=dict(title='Date'), yaxis=dict(title='Daily Sales'))

fig = go.Figure(data=[daily_sales_sp], layout=layout)

iplot(fig)
daily_sales_by_store = train.groupby(['date', 'shop_id'], axis=0, as_index=False)['sales'].sum()

daily_sales_by_store_sp = []

stores = np.sort(train.shop_id.unique())

for store in stores[26:36] :

    dummy = daily_sales_by_store[daily_sales_by_store.shop_id == store]

    daily_sales_by_store_sp.append(go.Scatter(x=dummy.date, y=dummy.sales, name='Store %s' % store))

    

layout = go.Layout(title='Daily Sales by Store', xaxis=dict(title='Date'), yaxis=dict(title='Sales'))

fig = go.Figure(data=daily_sales_by_store_sp, layout=layout)

iplot(fig)
daily_sales_by_item = train.groupby(['date', 'item_id'], as_index=False, axis=0)['sales'].sum()

daily_sales_by_item = daily_sales_by_item.sort_values('date', axis=0)



items = train.item_id.unique()

daily_sales_by_item_sp = []

for item in items[450:550] :

    dummy = daily_sales_by_item[daily_sales_by_item.item_id == item]

    daily_sales_by_item_sp.append(go.Scatter(x=dummy.date, y=dummy.sales, name=('item %s' %item)))

    

layout = go.Layout(title='Daily sales by item', xaxis=dict(title='Date'), yaxis=dict(title='sales'))

fig = go.Figure(data=daily_sales_by_item_sp, layout=layout)

iplot(fig)
test = pd.read_csv(bpath + 'test.csv')

print(test.shape)

test.head()
df_train = train.groupby([train.date.apply(lambda x: x.strftime('%Y-%m')), 'item_id', 'shop_id']).sum().reset_index()

df_train = df_train[['date','item_id','shop_id','item_cnt_day']]

df_train = df_train.pivot_table(index=['item_id','shop_id'], columns='date',

                                values='item_cnt_day',fill_value=0).reset_index()

df_train.head()
df_train.info()
df_test = pd.merge(test, df_train, on=['item_id','shop_id'], how='left')

df_test = df_test.fillna(0)

df_test = df_test.drop(labels=['ID', 'shop_id', 'item_id'], axis=1)

df_test.head()
last_month = '2015-12'

y_train = df_test[last_month]

x_train = df_test.drop(labels=[last_month], axis=1)

x_train = x_train.to_numpy()

y_train = y_train.to_numpy()

train_x, valid_x, train_y, valid_y = model_selection.train_test_split(x_train, y_train, 

                                                                      train_size=0.8, shuffle=True)

print(train_x.shape, train_y.shape, valid_x.shape, valid_y.shape)
x_test = df_test.drop(labels=['2013-01'], axis=1)

x_test = x_test.to_numpy()

print(x_test.shape)
def saleModel_mlp(input_shape) :

    x_input = Input(input_shape)

    x = Dense(64, activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(0.01),

              bias_regularizer=regularizers.l2(0.02))(x_input)

    x = Dense(32, activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(0.01),

              bias_regularizer=regularizers.l2(0.02))(x)

    x_out = Dense(1, activation=None)(x)

    

    model = Model(inputs=x_input, outputs=x_out, name='saleModel_mlp')

    

    return model
mlpModel = saleModel_mlp(np.shape(train_x[1,:]))

mlpModel.summary()
optim = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.99)

mlpModel.compile(optimizer=optim, metrics=['accuracy'], loss='mean_squared_error')
nepochs = 200

mlp_history = mlpModel.fit(x=train_x, y=train_y, validation_data=(valid_x, valid_y), epochs=nepochs, 

                           batch_size=512, verbose=1, shuffle=True, validation_split=0.0)
fig = plt.figure(figsize=(8,4))

plt.plot(range(nepochs), mlp_history.history['loss'], 'r', label='train')

plt.plot(range(nepochs), mlp_history.history['val_loss'], 'b', label='valid')

plt.legend()

plt.title('multi-layer perceptron')

plt.xlabel('epochs')

plt.ylabel('loss');
def saleModel_lstm(input_shape) :

    x_input = Input(input_shape)

    x = LSTM(units=32, activation='tanh', recurrent_activation='sigmoid', use_bias=True, 

                        kernel_initializer='glorot_uniform', return_sequences=True)(x_input)

    x = LSTM(units=16, activation='tanh', recurrent_activation='sigmoid', use_bias=True, 

                        kernel_initializer='glorot_uniform', return_sequences=False)(x)

    x_out = Dense(1, activation=None)(x)

    

    model = Model(inputs=x_input, outputs=x_out, name='saleModel_lstm')

    

    return model
train_xx = train_x.reshape((train_x.shape[0], train_x.shape[1], 1))

valid_xx = valid_x.reshape((valid_x.shape[0], valid_x.shape[1], 1))

test_xx = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

print(train_xx.shape, valid_xx.shape, test_xx.shape)
lstmModel = saleModel_lstm(np.shape(train_xx[1,:, :]))

lstmModel.summary()
lstmModel.compile(optimizer=optim, loss='mean_squared_error', metrics=['accuracy'])

lstm_history = lstmModel.fit(x=train_xx, y=train_y, validation_data=(valid_xx, valid_y), verbose=1,

                            epochs=100, batch_size=1024, shuffle=True)
fig = plt.figure(figsize=(8,4))

plt.plot(range(100), lstm_history.history['loss'], 'r', label='train')

plt.plot(range(100), lstm_history.history['val_loss'], 'b', label='valid')

plt.legend()

plt.title('LSTM')

plt.xlabel('epochs')

plt.ylabel('loss');
def saleModel_cnn(input_shape) :

    x_input = Input(input_shape)

    x = Conv1D(filters=64, padding='valid', strides=1, kernel_size=3)(x_input)

    x = Activation('relu')(x)

    x = MaxPooling1D(pool_size=2, strides=None, padding='valid')(x)

    

    x = Conv1D(filters=32, padding='valid', strides=1, kernel_size=3)(x)

    x = Activation('relu')(x)

    x = MaxPooling1D(pool_size=2, strides=None, padding='valid')(x)

    

    x = Flatten()(x)

    x_out = Dense(1, activation=None)(x)

    

    model = Model(inputs=x_input, outputs=x_out, name='saleModel_cnn')

    

    return model
cnnModel = saleModel_cnn(np.shape(train_xx[1,:,:]))

cnnModel.summary()
cnnModel.compile(optimizer=optim, loss='mse', metrics=['accuracy'])

cnn_history = cnnModel.fit(x=train_xx, y=train_y, validation_data=(valid_xx, valid_y), verbose=1,

                          epochs=nepochs, batch_size=512, shuffle=True)
fig = plt.figure(figsize=(5,4))

plt.plot(range(nepochs), cnn_history.history['loss'], 'r', label='train')

plt.plot(range(nepochs), cnn_history.history['val_loss'], 'b', label='valid')

plt.legend()

plt.xlabel('epochs')

plt.ylabel('loss')

plt.title('1D CNN');
mlp_pred = mlpModel.predict(x_test)

lstm_pred = lstmModel.predict(test_xx)

cnn_pred = cnnModel.predict(test_xx)

print(mlp_pred, lstm_pred, cnn_pred)
submission = pd.read_csv('sample_submission.csv')

submission.item_cnt_month = lstm_pred

submission.to_csv ('submission.csv', index = None, header = True)