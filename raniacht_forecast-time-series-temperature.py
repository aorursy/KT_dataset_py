import warnings
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from keras import optimizers
from keras.utils import plot_model
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error

'''
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot

%matplotlib inline
warnings.filterwarnings("ignore")
init_notebook_mode(connected=True)
'''
import tensorflow
from numpy.random import seed
tensorflow.random.set_seed(1)
seed(1)
dataset = pd.read_csv('../input/temperature/climate_hour.csv',index_col=0,header=0)
#dataset = dataset.sort_values('Date Time')
dataset.index = pd.to_datetime(dataset.index, format="%d.%m.%Y %H:%M:%S")
dataset.index
def series_to_supervised(data, window=1, lag=1, dropnan=True, simple=True, single=True):
    cols, names = list(), list()
    # Input sequence (t-n, ... t-1)
    for i in range(window, 0, -1):
        cols.append(data.shift(i))
        names += [('%s(t-%d)' % (col, i)) for col in data.columns]
    # Current timestep (t=0)
    
    cols.append(data)
    names += [('%s(t)' % (col)) for col in data.columns]
    if simple == False:
        # Target timestep (t=lag)
        if single == True:
            cols.append(data.shift(-lag))
            names += [('%s(t+%d)' % (col, lag)) for col in data.columns]
        if single == False:
            for j in range(1, lag+1, 1):
                cols.append(data.shift(-j))
                names += [('%s(t+%d)' % (col, j)) for col in data.columns]

    # Put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    agg.index = data.index
    # Drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
y_origine = dataset.loc['01.01.2015 01:00:00':'2017-01-01 00:00:00']
y_origine = y_origine.values
y_origine = y_origine[:,1]
values = dataset.values
i = 1
plt.figure()
for feature in range(0,14):
    plt.subplot(14, 1, i)
    plt.subplots_adjust(top = 20, bottom = 10.9)
    plt.plot(values[:,feature])
    plt.title(dataset.columns[feature], y=.8, loc='right')
    i += 1
plt.show()
print(np.argmin(values[:,11]))
print(values[57208:57211,11])
values[57208:57211,11] = 0
print(values[57208:57211,11])
i = 1
plt.figure()
for feature in range(0,14):
    plt.subplot(14, 1, i)
    plt.subplots_adjust(top = 20, bottom = 10.9)
    plt.plot(values[:,feature])
    plt.title(dataset.columns[feature], y=.8, loc='right')
    i += 1
plt.show()
print(np.argmin(values[:,12]))
print(values[57207:57211,12])
values[57207:57211,12] = 0
print(values[57207:57211,12])
i = 1
plt.figure()
for feature in range(0,14):
    plt.subplot(14, 1, i)
    plt.subplots_adjust(top = 20, bottom = 10.9)
    plt.plot(dataset.values[:,feature])
    plt.title(dataset.columns[feature], y=.8, loc='right')
    i += 1
plt.show()
normalized_y = MinMaxScaler(feature_range=(-1,1))
y_norm = values[:,1]
y_norm = y_norm.reshape(-1,1) 
y_norm = normalized_y.fit_transform(y_norm)
normalized = MinMaxScaler(feature_range=(-1,1))
data_normalized = normalized.fit_transform(values)
dataset_normalized = pd.DataFrame(data_normalized,columns=dataset.columns,index=dataset.index)
y_normaliz = pd.DataFrame(y_norm,index=dataset.index)
dataset_normalized.head()
window = 24
series = series_to_supervised(dataset_normalized, window=window)
series.head()
y = series['T (degC)(t)']
y.shape
series.drop(['p (mbar)(t)','T (degC)(t)', 'Tpot (K)(t)','Tdew (degC)(t)','rh (%)(t)','VPmax (mbar)(t)','VPact (mbar)(t)','VPdef (mbar)(t)','sh (g/kg)(t)','H2OC (mmol/mol)(t)','rho (g/m**3)(t)','wv (m/s)(t)','max. wv (m/s)(t)','wd (deg)(t)'], axis=1, inplace=True)
series.head()
series.shape
series  = series.sort_values('Date Time')
X_train = series.loc['2009-01-02 01:00:00':'01.01.2015 00:00:00']
X_valid = series.loc['01.01.2015 01:00:00':'2017-01-01 00:00:00']
Y_train = y_normaliz.loc['2009-01-02 01:00:00':'01.01.2015 00:00:00']
Y_valid = y_normaliz.loc['01.01.2015 01:00:00':'2017-01-01 00:00:00']
y_index_valid = Y_valid.index
print(X_train.shape,X_valid.shape,Y_train.shape,Y_valid.shape)
X_train = X_train.values
Y_train = Y_train.values
X_valid = X_valid.values
Y_valid = Y_valid.values
Y_train = Y_train.reshape(-1,1)
Y_valid = Y_valid.reshape(-1,1)
print(X_train.shape,X_valid.shape,Y_train.shape,Y_valid.shape)
timesteps = 24
ndim = 14
X_train_reformer = X_train.reshape(X_train.shape[0],timesteps,ndim)
X_valid_reformer = X_valid.reshape(X_valid.shape[0],timesteps,ndim)
Y_train = Y_train.reshape(Y_train.shape[0],)
Y_valid = Y_valid.reshape(Y_valid.shape[0],)
print(X_train_reformer.shape,X_valid_reformer.shape,Y_train.shape,Y_valid.shape)
resultat = pd.DataFrame({'temp_true':y_origine,'temp_pred':lstm_valid_pred_nn.ravel()},index=y_index_valid)
resultat.to_csv('resultat.csv')
mae_train_mlp, mae_val_mlp, mse_train_mlp, mse_val_mlp, mae_mlp, mse_mlp, mae_mlp_nn, mse_mlp_nn = [],[],[],[],[],[],[],[]
mlp_model = Sequential()
mlp_model.add(Dense(336, activation='relu', input_dim=X_train.shape[1]))
mlp_model.add(Dense(100, activation='relu'))
mlp_model.add(Dense(20, activation='relu'))
mlp_model.add(Dense(1))
mlp_model.compile(loss='mae',optimizer='adam', metrics=['mse'])
mlp_model.summary()
mlp_history = mlp_model.fit(X_train,Y_train, validation_data=(X_valid,Y_valid),epochs=10,verbose=2)
print(mlp_history.history['loss'][-1],mlp_history.history['val_loss'][-1])
mae_train_mlp.append(mlp_history.history['loss'][-1]) 
mae_val_mlp.append(mlp_history.history['val_loss'][-1]) 
mse_train_mlp.append(math.sqrt(mlp_history.history['mse'][-1])) 
mse_val_mlp.append(math.sqrt(mlp_history.history['val_mse'][-1]))
mlp_valid_pred = mlp_model.predict(X_valid)
MAE = mean_absolute_error(Y_valid, mlp_valid_pred, sample_weight=None, multioutput='uniform_average')
MSE = mean_squared_error(Y_valid, mlp_valid_pred, sample_weight=None, multioutput='uniform_average')
print(MAE, MSE)
mae_mlp.append(MAE)
mse_mlp.append(math.sqrt(MSE))
mlp_valid_pred_nn = normalized_y.inverse_transform(mlp_valid_pred)
MAE = mean_absolute_error(y_origine, mlp_valid_pred_nn, sample_weight=None, multioutput='uniform_average')
MSE = mean_squared_error(y_origine, mlp_valid_pred_nn, sample_weight=None, multioutput='uniform_average')
print(MAE,MSE)
mae_mlp_nn.append(MAE)
mse_mlp_nn.append(math.sqrt(MSE))
len(mae_mlp)
Y_valid_inv = normalized_y.inverse_transform(Y_valid.reshape(-1,1))
normalized_mlp_predictions = pd.DataFrame(Y_valid_inv, columns=['Temperature'])
normalized_mlp_predictions.index = y_index_valid 
normalized_mlp_predictions['Predicted Temperature'] = mlp_valid_pred_nn
normalized_mlp_predictions.head()
normalized_mlp_predictions.plot()
print(np.mean(mae_train_mlp),"/",min(mae_train_mlp))
print(np.mean(mae_val_mlp),"/",min(mae_val_mlp))
print(np.mean(mae_mlp),"/",min(mae_mlp))
print(np.mean(mae_mlp_nn),"/",min(mae_mlp_nn))
print(np.mean(mse_train_mlp),"/",min(mse_train_mlp))
print(np.mean(mse_val_mlp),"/",min(mse_val_mlp))
print(np.mean(mse_mlp),"/",min(mse_mlp))
print(np.mean(mse_mlp_nn),"/",min(mse_mlp_nn))
plt.plot(mlp_history.history['loss'])
plt.plot(mlp_history.history['val_loss'])
plt.title('model MLP')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
mae_train_cnn, mae_val_cnn, mse_train_cnn, mse_val_cnn, mae_cnn, mse_cnn, mae_cnn_nn, mse_cnn_nn = [],[],[],[],[],[],[],[]
model_cnn = Sequential()
model_cnn.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(timesteps,ndim)))
model_cnn.add(MaxPooling1D(pool_size=2))
model_cnn.add(Flatten())
model_cnn.add(Dense(24, activation='relu'))
model_cnn.add(Dense(1))
model_cnn.compile(loss='mae', optimizer='adam', metrics=['mse'])
model_cnn.summary()
cnn_history = model_cnn.fit(X_train_reformer,Y_train,validation_data=(X_valid_reformer,Y_valid),epochs=10,verbose=2)
print(cnn_history.history['loss'][-1],cnn_history.history['val_loss'][-1])
'''
mae_train_cnn.append(cnn_history.history['loss'][-1]) 
mae_val_cnn.append(cnn_history.history['val_loss'][-1]) 
mse_train_cnn.append(math.sqrt(cnn_history.history['mse'][-1])) 
mse_val_cnn.append(math.sqrt(cnn_history.history['val_mse'][-1]))
'''
cnn_valid_pred = model_cnn.predict(X_valid_reformer)
MAE = mean_absolute_error(Y_valid, cnn_valid_pred, sample_weight=None, multioutput='uniform_average')
MSE = mean_squared_error(Y_valid, cnn_valid_pred, sample_weight=None, multioutput='uniform_average')
print(MAE,MSE)
#mae_cnn.append(MAE)
#mse_cnn.append(math.sqrt(MSE))
cnn_valid_pred_nn = normalized_y.inverse_transform(cnn_valid_pred)
MAE = mean_absolute_error(y_origine, cnn_valid_pred_nn, sample_weight=None, multioutput='uniform_average')
MSE = mean_squared_error(y_origine, cnn_valid_pred_nn, sample_weight=None, multioutput='uniform_average')
print(MAE,MSE)
#mae_cnn_nn.append(MAE)
#mse_cnn_nn.append(math.sqrt(MSE))
normalized_cnn_predictions = pd.DataFrame(Y_valid_inv, columns=['Temperature'])
normalized_cnn_predictions.index = y_index_valid 
normalized_cnn_predictions['Predicted Temperature'] = cnn_valid_pred_nn
normalized_cnn_predictions.head()
normalized_cnn_predictions.plot()
len(mae_cnn_nn)
print(np.mean(mae_train_cnn),"/",min(mae_train_cnn))
print(np.mean(mae_val_cnn),"/",min(mae_val_cnn))
print(np.mean(mae_cnn),"/",min(mae_cnn))
print(np.mean(mae_cnn_nn),"/",min(mae_cnn_nn))
print(np.mean(mse_train_cnn),"/",min(mse_train_cnn))
print(np.mean(mse_val_cnn),"/",min(mse_val_cnn))
print(np.mean(mse_cnn),"/",min(mse_cnn))
print(np.mean(mse_cnn_nn),"/",min(mse_cnn_nn))
mae_train_lstm, mae_val_lstm, mse_train_lstm, mse_val_lstm, mae_lstm, mse_lstm, mae_lstm_nn, mse_lstm_nn = [],[],[],[],[],[],[],[]
lstm_model = Sequential()
lstm_model.add(LSTM(50, input_shape=(timesteps,ndim)))
lstm_model.add(Dense(1))
lstm_model.compile(loss='mae', optimizer='adam', metrics=['mse'])
lstm_model.summary()
lstm_history = lstm_model.fit(X_train_reformer,Y_train,validation_data=(X_valid_reformer,Y_valid),epochs=10,verbose=2)
plt.plot(lstm_history.history['loss'])
plt.plot(lstm_history.history['val_loss'])
plt.title('model LSTM')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
print(lstm_history.history['loss'][-1],lstm_history.history['val_loss'][-1])
print(math.sqrt(lstm_history.history['mse'][-1]),math.sqrt(lstm_history.history['val_mse'][-1]))
'''
mae_train_lstm.append(lstm_history.history['loss'][-1]) 
mae_val_lstm.append(lstm_history.history['val_loss'][-1]) 
mse_train_lstm.append(math.sqrt(lstm_history.history['mse'][-1])) 
mse_val_lstm.append(math.sqrt(lstm_history.history['val_mse'][-1]))
'''
lstm_valid_pred = lstm_model.predict(X_valid_reformer)
MAE = mean_absolute_error(Y_valid, lstm_valid_pred, sample_weight=None, multioutput='uniform_average')
RMSE = math.sqrt(mean_squared_error(Y_valid, lstm_valid_pred, sample_weight=None, multioutput='uniform_average'))
print(MAE,RMSE)
#mae_lstm.append(MAE)
#mse_lstm.append(math.sqrt(MSE))
lstm_valid_pred_nn = normalized_y.inverse_transform(lstm_valid_pred)
MAE = mean_absolute_error(y_origine, lstm_valid_pred_nn, sample_weight=None, multioutput='uniform_average')
RMSE = math.sqrt(mean_squared_error(y_origine, lstm_valid_pred_nn, sample_weight=None, multioutput='uniform_average'))
print(MAE, RMSE)
#mae_lstm_nn.append(MAE)
#mse_lstm_nn.append(math.sqrt(MSE))
import matplotlib.dates as mdates
fig = plt.figure(figsize=(14, 5))
format_date = mdates.DateFormatter('%d/%m/%y')
ax = fig.add_subplot(111)
ax.plot(y_index_valid,y_origine, label='observed')
ax.plot(y_index_valid,lstm_valid_pred_nn.ravel(), label='predicted')
ax.xaxis.set_major_formatter(format_date)
ax.set_title("Prediction vs Observation")
ax.set_xlabel("data")
ax.set_ylabel("Temperature (C°)")
ax.legend()
ax.tick_params(axis='x', rotation=70)
plt.show()
normalized_lstm_predictions = pd.DataFrame(Y_valid_inv, columns=['Temperature'])
normalized_lstm_predictions.index = y_index_valid 
normalized_lstm_predictions['Predicted Temperature'] = lstm_valid_pred_nn
normalized_lstm_predictions.head()
normalized_lstm_predictions.plot()
len(mae_lstm_nn)
print(np.mean(mae_train_lstm),"/",min(mae_train_lstm))
print(np.mean(mae_val_lstm),"/",min(mae_val_lstm))
print(np.mean(mae_lstm),"/",min(mae_lstm))
print(np.mean(mae_lstm_nn),"/",min(mae_lstm_nn))
print(np.mean(mse_train_lstm),"/",min(mse_train_lstm))
print(np.mean(mse_val_lstm),"/",min(mse_val_lstm))
print(np.mean(mse_lstm),"/",min(mse_lstm))
print(np.mean(mse_lstm_nn),"/",min(mse_lstm_nn))
subsequences = 14
timesteps = 24 #X_train_series.shape[1]//subsequences
X_train_series_sub = X_train_reformer.reshape((X_train_reformer.shape[0], subsequences, timesteps, 1))
X_valid_series_sub = X_valid_reformer.reshape((X_valid_reformer.shape[0], subsequences, timesteps, 1))
print('Train set shape', X_train_series_sub.shape)
print('Validation set shape', X_valid_series_sub.shape)
mae_train_cnn_lstm, mae_val_cnn_lstm, mse_train_cnn_lstm, mse_val_cnn_lstm, mae_cnn_lstm, mse_cnn_lstm, mae_cnn_lstm_nn, mse_cnn_lstm_nn = [],[],[],[],[],[],[],[]
model_cnn_lstm = Sequential()
model_cnn_lstm.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, X_train_series_sub.shape[2], X_train_series_sub.shape[3])))
model_cnn_lstm.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model_cnn_lstm.add(TimeDistributed(Flatten()))
model_cnn_lstm.add(LSTM(50, activation='relu'))
model_cnn_lstm.add(Dense(1))
model_cnn_lstm.compile(loss='mae', optimizer='adam', metrics=['mse'])
cnn_lstm_history = model_cnn_lstm.fit(X_train_series_sub, Y_train, validation_data=(X_valid_series_sub, Y_valid), epochs=10, verbose=2)
print(cnn_lstm_history.history['loss'][-1],cnn_lstm_history.history['val_loss'][-1])
mae_train_cnn_lstm.append(cnn_lstm_history.history['loss'][-1]) 
mae_val_cnn_lstm.append(cnn_lstm_history.history['val_loss'][-1]) 
mse_train_cnn_lstm.append(math.sqrt(cnn_lstm_history.history['mse'][-1])) 
mse_val_cnn_lstm.append(math.sqrt(cnn_lstm_history.history['val_mse'][-1]))
cnn_lstm_valid_pred = model_cnn_lstm.predict(X_valid_series_sub)
MAE = mean_absolute_error(Y_valid, cnn_lstm_valid_pred, sample_weight=None, multioutput='uniform_average')
MSE = mean_squared_error(Y_valid, cnn_lstm_valid_pred, sample_weight=None, multioutput='uniform_average')
print(MAE,MSE)
#mae_cnn_lstm.append(MAE)
#mse_cnn_lstm.append(math.sqrt(MSE))
cnn_lstm_valid_pred_nn = normalized_y.inverse_transform(cnn_lstm_valid_pred)
MAE = mean_absolute_error(y_origine, cnn_lstm_valid_pred_nn, sample_weight=None, multioutput='uniform_average')
MSE = mean_squared_error(y_origine, cnn_lstm_valid_pred_nn, sample_weight=None, multioutput='uniform_average')
print(MAE,MSE)
#mae_cnn_lstm_nn.append(MAE)
#mse_cnn_lstm_nn.append(math.sqrt(MSE))
normalized_cnn_lstm_predictions = pd.DataFrame(Y_valid_inv, columns=['Temperature'])
normalized_cnn_lstm_predictions.index = y_index_valid 
normalized_cnn_lstm_predictions['Predicted Temperature'] = cnn_lstm_valid_pred_nn
normalized_cnn_lstm_predictions.head()
normalized_cnn_lstm_predictions.plot()
len(mse_cnn_lstm_nn)
print(np.mean(mae_train_cnn_lstm),"/",min(mae_train_cnn_lstm))
print(np.mean(mae_val_cnn_lstm),"/",min(mae_val_cnn_lstm))
print(np.mean(mae_cnn_lstm),"/",min(mae_cnn_lstm))
print(np.mean(mae_cnn_lstm_nn),"/",min(mae_cnn_lstm_nn))
print(np.mean(mse_train_cnn_lstm),"/",min(mse_train_cnn_lstm))
print(np.mean(mse_val_cnn_lstm),"/",min(mse_val_cnn_lstm))
print(np.mean(mse_cnn_lstm),"/",min(mse_cnn_lstm))
print(np.mean(mse_cnn_lstm_nn),"/",min(mse_cnn_lstm_nn))
window=24
lag=24
series_single = series_to_supervised(dataset_normalized, window=window, lag=lag, simple=False)
series_single.head()
print(series_single.shape)
y_single = series_single['T (degC)(t+24)']
series_single.drop(['p (mbar)(t+24)','T (degC)(t+24)', 'Tpot (K)(t+24)','Tdew (degC)(t+24)','rh (%)(t+24)','VPmax (mbar)(t+24)','VPact (mbar)(t+24)','VPdef (mbar)(t+24)','sh (g/kg)(t+24)','H2OC (mmol/mol)(t+24)','rho (g/m**3)(t+24)','wv (m/s)(t+24)','max. wv (m/s)(t+24)','wd (deg)(t+24)'], axis=1, inplace=True)
series_single.head()
print(y_single.shape)
series_single  = series_single.sort_values('Date Time')
X_train_single = series_single.loc['02-01-2009 01:00:00':'01.01.2015 00:00:00']
X_valid_single = series_single.loc['01.01.2015 01:00:00':'31-12-2016 00:00:00']
Y_train_single = y_normaliz.loc['02-01-2009 01:00:00':'01.01.2015 00:00:00']
Y_valid_single = y_normaliz.loc['01.01.2015 01:00:00':'31-12-2016 00:00:00']
y_index_valid_single = Y_valid_single.index
print(X_train_single.shape,X_valid_single.shape,Y_train_single.shape,Y_valid_single.shape)
X_train_single = X_train_single.values
Y_train_single = Y_train_single.values
X_valid_single = X_valid_single.values
Y_valid_single = Y_valid_single.values
Y_train_single = Y_train_single.reshape(-1,1)
Y_valid_single = Y_valid_single.reshape(-1,1)
print(X_train_single.shape,X_valid_single.shape,Y_train_single.shape,Y_valid_single.shape)
timesteps = 25
ndim = 14
X_train_single_reformer = X_train_single.reshape(X_train_single.shape[0],timesteps,ndim)
X_valid_single_reformer = X_valid_single.reshape(X_valid_single.shape[0],timesteps,ndim)
Y_train_single = Y_train_single.reshape(Y_train_single.shape[0],)
Y_valid_single = Y_valid_single.reshape(Y_valid_single.shape[0],)
print(X_train_single_reformer.shape,X_valid_single_reformer.shape,Y_train_single.shape,Y_valid_single.shape)
lstm_model_single = Sequential()
lstm_model_single.add(LSTM(50, input_shape=(25,14)))
lstm_model_single.add(Dense(1))
lstm_model_single.compile(loss='mae', optimizer='adam', metrics=['mse'])
lstm_model_single.summary()
lstm_history_single = lstm_model_single.fit(X_train_single_reformer,Y_train_single,validation_data=(X_valid_single_reformer,Y_valid_single),epochs=10,verbose=2)
plt.plot(lstm_history_single.history['loss'])
plt.plot(lstm_history_single.history['val_loss'])
plt.title('model LSTM Single Step')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
print(lstm_history_single.history['loss'][-1],lstm_history_single.history['val_loss'][-1])
print(math.sqrt(lstm_history_single.history['mse'][-1]),math.sqrt(lstm_history_single.history['val_mse'][-1]))
lstm_valid_single_pred = lstm_model_single.predict(X_valid_single_reformer)
MAE = mean_absolute_error(Y_valid_single, lstm_valid_single_pred, sample_weight=None, multioutput='uniform_average')
RMSE = math.sqrt(mean_squared_error(Y_valid_single, lstm_valid_single_pred, sample_weight=None, multioutput='uniform_average'))
print(MAE,RMSE)
lstm_valid_single_pred_nn = normalized_y.inverse_transform(lstm_valid_single_pred)
MAE = mean_absolute_error(y_origine[24:], lstm_valid_single_pred_nn, sample_weight=None, multioutput='uniform_average')
RMSE = math.sqrt(mean_squared_error(y_origine[24:], lstm_valid_single_pred_nn, sample_weight=None, multioutput='uniform_average'))
print(MAE, RMSE)
import matplotlib.dates as mdates
fig = plt.figure(figsize=(14, 5))
format_date = mdates.DateFormatter('%d/%m/%y')
ax = fig.add_subplot(111)
ax.plot(y_index_valid_single,y_origine[24:], label='observed')
ax.plot(y_index_valid_single,lstm_valid_single_pred_nn.ravel(), label='predicted')
ax.xaxis.set_major_formatter(format_date)
ax.set_title("Prediction vs Observation")
ax.set_xlabel("data")
ax.set_ylabel("Temperature (C°)")
ax.legend()
ax.tick_params(axis='x', rotation=70)
plt.show()
window = 24
lag =24
series_multi = series_to_supervised(dataset_normalized, window=window, lag=lag, simple=False, single=False)
series_multi.head()
print(series_multi.shape)
features = [('T (degC)(t+%d)' % (i)) for i in range(1, lag+1)]
y_multi = series_multi[features]
plus = [('%s(t+%d)' % (f,k)) for f in dataset.columns for k in range(1,lag+1)]
series_multi.drop(plus, axis=1, inplace=True)
series_multi.head()
print(y_multi.shape)
series_multi  = series_multi.sort_values('Date Time')
X_train_multi = series_multi.loc['02-01-2009 01:00:00':'01.01.2015 00:00:00']
X_valid_multi = series_multi.loc['01.01.2015 01:00:00':'31-12-2016 00:00:00']
Y_train_multi = y_multi.loc['02-01-2009 01:00:00':'01.01.2015 00:00:00']
Y_valid_multi = y_multi.loc['01.01.2015 01:00:00':'31-12-2016 00:00:00']
y_index_valid_multi = Y_valid_multi.index
print(X_train_multi.shape,X_valid_multi.shape,Y_train_multi.shape,Y_valid_multi.shape)
X_train_multi = X_train_multi.values
Y_train_multi = Y_train_multi.values
X_valid_multi = X_valid_multi.values
Y_valid_multi = Y_valid_multi.values
Y_train_multi = Y_train_multi.reshape(-1,24)
Y_valid_multi = Y_valid_multi.reshape(-1,24)
print(X_train_multi.shape,X_valid_multi.shape,Y_train_multi.shape,Y_valid_multi.shape)
timesteps = 25
ndim = 14
X_train_multi_reformer = X_train_multi.reshape(X_train_multi.shape[0],timesteps,ndim)
X_valid_multi_reformer = X_valid_multi.reshape(X_valid_multi.shape[0],timesteps,ndim)
Y_train_multi = Y_train_multi.reshape(Y_train_multi.shape[0],24)
Y_valid_multi = Y_valid_multi.reshape(Y_valid_multi.shape[0],24)
print(X_train_multi_reformer.shape,X_valid_multi_reformer.shape,Y_train_multi.shape,Y_valid_multi.shape)
lstm_model_multi = Sequential()
lstm_model_multi.add(LSTM(50, input_shape=(25,14)))
lstm_model_multi.add(Dense(24))
lstm_model_multi.compile(loss='mae', optimizer='adam', metrics=['mse'])
lstm_model_multi.summary()
lstm_history_multi = lstm_model_multi.fit(X_train_multi_reformer,Y_train_multi,validation_data=(X_valid_multi_reformer,Y_valid_multi),epochs=10,verbose=2)
plt.plot(lstm_history_multi.history['loss'])
plt.plot(lstm_history_multi.history['val_loss'])
plt.title('model LSTM Multi Step')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
print(lstm_history_multi.history['loss'][-1],lstm_history_multi.history['val_loss'][-1])
print(math.sqrt(lstm_history_multi.history['mse'][-1]),math.sqrt(lstm_history_multi.history['val_mse'][-1]))
lstm_valid_multi_pred = lstm_model_multi.predict(X_valid_multi_reformer)
MAE = mean_absolute_error(Y_valid_multi, lstm_valid_multi_pred, sample_weight=None, multioutput='uniform_average')
RMSE = math.sqrt(mean_squared_error(Y_valid_multi, lstm_valid_multi_pred, sample_weight=None, multioutput='uniform_average'))
print(MAE,RMSE)
y_true = normalized_y.inverse_transform(Y_valid_multi)
lstm_valid_multi_pred_nn = normalized_y.inverse_transform(lstm_valid_multi_pred)
MAE = mean_absolute_error(y_true, lstm_valid_multi_pred_nn, sample_weight=None, multioutput='uniform_average')
RMSE = math.sqrt(mean_squared_error(y_true, lstm_valid_multi_pred_nn, sample_weight=None, multioutput='uniform_average'))
print(MAE, RMSE)
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
fig = plt.figure(figsize=(14, 5))
format_date = mdates.DateFormatter('%d/%m/%Y')
ax = fig.add_subplot(111)
ax.plot(y_index_valid_multi,lstm_valid_multi_pred_nn, label='predicted', color='green')
ax.plot(y_index_valid_multi,y_true, label='observed', color='red')
ax.xaxis.set_major_formatter(format_date)
ax.set_title("Prediction vs Observation")
ax.set_xlabel("data")
ax.set_ylabel("Temperature (C°)")
#ax.legend(lines[:2], ['first', 'second']);
red_patch = mpatches.Patch(color='red', label='observed')
vert_patch = mpatches.Patch(color='green', label='predicted')
ax.legend(handles=[red_patch, vert_patch], loc='lower left', frameon=False, ncol=2)
ax.tick_params(axis='x', rotation=70)
plt.show()