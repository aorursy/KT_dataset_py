!pip install chart_studio
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
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot

%matplotlib inline
warnings.filterwarnings("ignore")
init_notebook_mode(connected=True)

# Set seeds to make the experiment more reproducible.
import tensorflow as tf
from numpy.random import seed
tf.random.set_seed(1)
seed(1)
import numpy as np 
data = pd.read_csv('../input/climate-hour/climate_hour.csv', parse_dates=['Date Time'],index_col = 0, header=0)
data = data.sort_values(['Date Time'])
data.head() 
new_data = data['T (degC)']
#new_data = new_data.array.reshape(-1, 1 )                                 
new_data = pd.DataFrame({'Date Time': data.index, 'T (degC)':new_data.values})
new_data = new_data.set_index(['Date Time'])
new_data.head()
from sklearn.preprocessing import MinMaxScaler
temp_scaler = MinMaxScaler()
temp_scaler.fit(new_data) 
normalized_temp = temp_scaler.transform(new_data) 
normalized_temp = pd.DataFrame(normalized_temp, columns=['Normalized Temperature'])
normalized_temp.index = new_data.index
normalized_temp.head()
data_scaler = MinMaxScaler()
data_scaler.fit(data) 
normalized_data = data_scaler.transform(data) 

normalized_df = pd.DataFrame(normalized_data, columns=['p (mbar)','T (degC)','Tpot (K)','Tdew (degC)','rh (%)','VPmax (mbar)','VPact (mbar)','VPdef (mbar)','sh (g/kg)','H2OC (mmol/mol)','rho (g/m**3)','wv (m/s)','max. wv (m/s)','wd (deg)'])
normalized_df = normalized_df.set_index(data.index)
normalized_df.head()
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
window = 24
series = series_to_supervised(normalized_df, window=window)
series.head()
print(series.values.shape)
print(np.isnan(series.values).any())
# Label #52517
#labels_col = 'T (degC)(t+%d)' % lag_size
labels_col = 'T (degC)(t)'
labels = series[labels_col]
series = series.drop(labels_col, axis=1)
X_train = series['2009-01-02 01:00:00':'01.01.2015 00:00:00']
X_valid = series['01.01.2015 00:00:00':'2017-01-01 00:00:00'] 
Y_train = labels['2009-01-02 01:00:00':'01.01.2015 00:00:00']
Y_valid = labels['01.01.2015 00:00:00':'2017-01-01 00:00:00']
print('Train set shape', X_train.shape)
print('Validation set shape', X_valid.shape)
epochs = 10
batch = 254
lr = 0.0003
adam = optimizers.Adam(lr)
import time
name = "model-mlp{}".format(int(time.time()))
model_mlp = Sequential()
model_mlp.add(Dense(100, activation='relu', input_dim=X_train.shape[1]))
model_mlp.add(Dense(100, activation='relu', input_dim=X_train.shape[1]))
model_mlp.add(Dense(1))
model_mlp.compile(loss='mae', optimizer=adam, metrics=['mse','accuracy']) 
model_mlp.summary()
#Saving the model :
model_mlp.save(name)   
mlp_history = model_mlp.fit(X_train.values, Y_train, validation_data=(X_valid.values, Y_valid), epochs=epochs, verbose=2)
#Saving history in a csv file :
hist_df = pd.DataFrame(mlp_history.history) 
hist_csv_file = 'mlp-history-{}.csv'.format(int(time.time()))
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)             
import matplotlib.pyplot as plt 
mlp_train_loss = mlp_history.history['loss']
mlp_test_loss = mlp_history.history['val_loss']

epoch_count = range(1, len(mlp_train_loss)+1)

plt.plot(epoch_count, mlp_train_loss)
plt.plot(epoch_count, mlp_test_loss)
plt.title('loss history')
plt.legend(['train', 'validation'])
plt.xlabel('Epoch')
plt.xlabel('Loss')
plt.show()
#ajouté :
mlp_train_pred = model_mlp.predict(X_train.values)
mlp_valid_pred = model_mlp.predict(X_valid.values)

print('Train rmse (avec normalisation):', np.sqrt(mean_squared_error(Y_train, mlp_train_pred)))
print('Validation rmse (avec normalisation):', np.sqrt(mean_squared_error(Y_valid, mlp_valid_pred)))
from sklearn.metrics import mean_absolute_error
print('Train mae (avec normalisation):', mean_absolute_error(Y_train, mlp_train_pred))
print('Validation mae (avec normalisation):', mean_absolute_error(Y_valid, mlp_valid_pred))
normalized_mlp_predictions = pd.DataFrame(Y_valid.values, columns=['Temperature'])
normalized_mlp_predictions.index = X_valid.index 
normalized_mlp_predictions['Predicted Temperature'] = mlp_valid_pred
normalized_mlp_predictions.head()
normalized_mlp_predictions.tail()
normalized_mlp_predictions.plot()
#y_valid = Y_valid.values
#Y_valid_inverse = scaler.inverse_transform(y_valid.reshape(-1,1))
y_val_dataset = np.zeros(shape=(len(mlp_valid_pred), 14) )
y_val_dataset[:,1] = normalized_mlp_predictions['Temperature'] 
y_val_inv = data_scaler.inverse_transform(y_val_dataset)[:,1]
print(y_val_inv)
y_val = new_data['01.01.2015 00:00:00':'2016-12-31 23:00:00']
print(y_val)
print(mlp_valid_pred)
print(normalized_mlp_predictions['Predicted Temperature'])
pred_dataset = np.zeros(shape=(len(mlp_valid_pred), 14) )
pred_dataset[:,1] = mlp_valid_pred.reshape(17470) #normalized_mlp_predictions['Predicted Temperature'] 
y_pred_inv = data_scaler.inverse_transform(pred_dataset)[:,1]
mlp_predictions = pd.DataFrame(y_val.values, columns=['True Temperature'])
#mlp_predictions['Date Time'] = dates[52565:] 
mlp_predictions.index = y_val.index #new_data['31.12.2014 23:00:00':'2017-01-01 00:00:00'].index #X_valid.index 
mlp_predictions['Predicted Temperature'] = y_pred_inv
print(mlp_predictions)
mlp_predictions.to_csv('new-mlp-predictions.csv')
from sklearn.metrics import mean_absolute_error
print('Validation mae (sans normalisation):', mean_absolute_error(mlp_predictions['True Temperature'], mlp_predictions['Predicted Temperature']))
mlp_predictions.plot()
import matplotlib.pyplot as plt 
mlp_true_temp = mlp_predictions['True Temperature']
time_stamp = mlp_predictions.index
plt.plot(time_stamp, mlp_true_temp)
plt.xlabel('Time')
plt.ylabel('True Temperature C°')
plt.show() 
import matplotlib.pyplot as plt 
mlp_pred_temp = mlp_predictions['Predicted Temperature']
time_stamp = mlp_predictions.index
plt.plot(time_stamp, mlp_pred_temp)
plt.xlabel('Time')
plt.xlabel('Predicted Temperature C°')
plt.show() 
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
model_cnn.compile(loss='mae', optimizer=adam, metrics=['mse','accuracy']) #'mse'
model_cnn.summary()
cnn_history = model_cnn.fit(X_train_series, Y_train, validation_data=(X_valid_series, Y_valid), epochs=epochs, verbose=2)
#Saving history in a csv file :
cnn_hist_df = pd.DataFrame(cnn_history.history) 
cnn_hist_csv_file = 'cnn-history-{}.csv'.format(int(time.time()))
with open(cnn_hist_csv_file, mode='w') as f:
    cnn_hist_df.to_csv(f) 
import matplotlib.pyplot as plt 
cnn_train_loss = cnn_history.history['loss']
cnn_test_loss = cnn_history.history['val_loss']

epoch_count = range(1, len(cnn_train_loss)+1)

plt.plot(epoch_count, cnn_train_loss)
plt.plot(epoch_count, cnn_test_loss)
plt.title('loss history')
plt.legend(['train', 'validation'])
plt.xlabel('Epoch')
plt.xlabel('Accuracy')
plt.show()
#Normalized predictions:
cnn_train_pred = model_cnn.predict(X_train_series)
cnn_valid_pred = model_cnn.predict(X_valid_series)

print('Train rmse (avec normalisation):', np.sqrt(mean_squared_error(Y_train, cnn_train_pred)))
print('Validation rmse (avec normalisation):', np.sqrt(mean_squared_error(Y_valid, cnn_valid_pred)))
from sklearn.metrics import mean_absolute_error
print('Train mae (avec normalisation):', mean_absolute_error(Y_train, cnn_train_pred))
print('Validation mae (avec normalisation):', mean_absolute_error(Y_valid, cnn_valid_pred))
normalized_cnn_predictions = pd.DataFrame(Y_valid.values, columns=['Temperature'])
normalized_cnn_predictions.index = X_valid.index 
normalized_cnn_predictions['Predicted Temperature'] = cnn_valid_pred
normalized_cnn_predictions.head()
normalized_cnn_predictions.tail()
normalized_cnn_predictions.plot()
print(cnn_valid_pred)
print(normalized_cnn_predictions['Temperature'] )
y_val_cnn_dataset = np.zeros(shape=(len(cnn_valid_pred), 14) )
y_val_cnn_dataset[:,1] = normalized_cnn_predictions['Temperature']
y_val_inv_cnn = data_scaler.inverse_transform(y_val_cnn_dataset)[:,1]
print(y_val_inv_cnn)
pred_cnn_dataset = np.zeros(shape=(len(cnn_valid_pred), 14) )
pred_cnn_dataset[:,1] = cnn_valid_pred.reshape(17470)
y_pred_inv_cnn = data_scaler.inverse_transform(pred_cnn_dataset)[:,1]
cnn_predictions = pd.DataFrame(y_val.values, columns=['True Temperature'])
cnn_predictions.index = y_val.index 
cnn_predictions['Predicted Temperature'] = y_pred_inv_cnn
print(cnn_predictions)
cnn_predictions.to_csv('new-cnn-predictions.csv')
from sklearn.metrics import mean_absolute_error
print('Validation mae (sans normalisation):', mean_absolute_error(cnn_predictions['True Temperature'], cnn_predictions['Predicted Temperature']))
cnn_predictions.plot()
import matplotlib.pyplot as plt 
cnn_true_temp = cnn_predictions['True Temperature']
time_stamp = cnn_predictions.index
plt.plot(time_stamp, cnn_true_temp)
plt.xlabel('Time')
plt.ylabel('True Temperature C°')
plt.show() 
import matplotlib.pyplot as plt 
cnn_pred_temp = cnn_predictions['Predicted Temperature']
time_stamp = cnn_predictions.index
plt.plot(time_stamp, cnn_pred_temp)
plt.xlabel('Time')
plt.xlabel('Predicted Temperature C°')
plt.show() 
X_train_series = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
X_valid_series = X_valid.values.reshape((X_valid.shape[0], X_valid.shape[1], 1))
print('Train set shape', X_train_series.shape)
print('Validation set shape', X_valid_series.shape)
from keras.regularizers import l1
# instantiate regularizer
reg = l1(0.001)
model_lstm = Sequential()
model_lstm.add(LSTM(50, activation='relu', input_shape=(X_train_series.shape[1], X_train_series.shape[2]),activity_regularizer=l1(0.001)))
model_lstm.add(Dense(1)) #, activation='softmax')  #(Dense(5, activation='softmax')) #, activation='relu'
model_lstm.compile(loss='mae', optimizer=adam, metrics=['mse','accuracy']) #'mse'
model_lstm.summary()
print(X_train_series.shape)
print(X_train_series.shape[1], X_train_series.shape[2])
print(X_train_series.shape[1], X_train_series.shape[2])
lstm_history = model_lstm.fit(X_train_series, Y_train, validation_data=(X_valid_series, Y_valid), epochs=10, verbose=2)
print(np.isnan(Y_train).any())
#Saving history in a csv file :
lstm_hist_df = pd.DataFrame(lstm_history.history) 
lstm_hist_csv_file = 'lstm-history-{}.csv'.format(int(time.time()))
with open(lstm_hist_csv_file, mode='w') as f:
    lstm_hist_df.to_csv(f) 
import matplotlib.pyplot as plt 
lstm_train_loss = lstm_history.history['loss']
lstm_test_loss = lstm_history.history['val_loss']

epoch_count = range(1, len(lstm_train_loss)+1)

plt.plot(epoch_count, lstm_train_loss)
plt.plot(epoch_count, lstm_test_loss)
plt.title('loss history')
plt.legend(['train', 'validation'])
plt.xlabel('Epoch')
plt.xlabel('Accuracy')
plt.show()
#Normalized predictions:
lstm_train_pred = model_lstm.predict(X_train_series)
lstm_valid_pred = model_lstm.predict(X_valid_series)

print('Train rmse (avec normalisation):', np.sqrt(mean_squared_error(Y_train, lstm_train_pred)))
print('Validation rmse (avec normalisation):', np.sqrt(mean_squared_error(Y_valid, lstm_valid_pred)))
from sklearn.metrics import mean_absolute_error
print('Train mae (avec normalisation):', mean_absolute_error(Y_train, lstm_train_pred))
print('Validation mae (avec normalisation):', mean_absolute_error(Y_valid, lstm_valid_pred))
normalized_lstm_predictions = pd.DataFrame(Y_valid.values, columns=['Temperature'])
normalized_lstm_predictions.index = X_valid.index 
normalized_lstm_predictions['Predicted Temperature'] = lstm_valid_pred
normalized_lstm_predictions.head()
normalized_lstm_predictions.tail()
normalized_lstm_predictions.plot()
print(lstm_valid_pred)
print(normalized_lstm_predictions['Temperature'] )
y_val_lstm_dataset = np.zeros(shape=(len(lstm_valid_pred), 14) )
y_val_lstm_dataset[:,1] = normalized_lstm_predictions['Temperature']
y_val_inv_lstm = data_scaler.inverse_transform(y_val_lstm_dataset)[:,1]
print(y_val_inv_lstm)
pred_lstm_dataset = np.zeros(shape=(len(lstm_valid_pred), 14) )
pred_lstm_dataset[:,1] = lstm_valid_pred.reshape(17470)
y_pred_inv_lstm = data_scaler.inverse_transform(pred_lstm_dataset)[:,1]
lstm_predictions = pd.DataFrame(y_val.values, columns=['True Temperature'])
lstm_predictions.index = y_val.index 
lstm_predictions['Predicted Temperature'] = y_pred_inv_lstm
print(lstm_predictions)
lstm_predictions.to_csv('new-lstm-predictions.csv') 
from sklearn.metrics import mean_absolute_error
print('Validation mae (sans normalisation):', mean_absolute_error(lstm_predictions['True Temperature'], lstm_predictions['Predicted Temperature']))
lstm_predictions.plot()
import matplotlib.pyplot as plt 
lstm_true_temp = lstm_predictions['True Temperature']
time_stamp = lstm_predictions.index
plt.plot(time_stamp, lstm_true_temp)
plt.xlabel('Time')
plt.ylabel('True Temperature C°')
plt.show() 
import matplotlib.pyplot as plt 
lstm_pred_temp = lstm_predictions['Predicted Temperature']
time_stamp = lstm_predictions.index
plt.plot(time_stamp, lstm_pred_temp)
plt.xlabel('Time')
plt.xlabel('Predicted Temperature C°')
plt.show() 
X_train
#original
'''
subsequences = 5
timesteps = X_train_series.shape[1]//subsequences
X_train_series_sub = X_train_series.reshape((X_train_series.shape[0], subsequences, timesteps, 1))
X_valid_series_sub = X_valid_series.reshape((X_valid_series.shape[0], subsequences, timesteps, 1))
print('Train set shape', X_train_series_sub.shape)
print('Validation set shape', X_valid_series_sub.shape)
'''
subsequences = 1
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
model_cnn_lstm.compile(loss='mae', optimizer=adam, metrics=['mse','accuracy']) #'mse'
cnn_lstm_history = model_cnn_lstm.fit(X_train_series_sub, Y_train, validation_data=(X_valid_series_sub, Y_valid), epochs=epochs, verbose=2)
#Saving history in a csv file :
cnn_lstm_hist_df = pd.DataFrame(cnn_lstm_history.history) 
cnn_lstm_hist_csv_file = 'cnn-lstm-history-{}.csv'.format(int(time.time()))
with open(cnn_lstm_hist_csv_file, mode='w') as f:
    cnn_lstm_hist_df.to_csv(f) 
import matplotlib.pyplot as plt 
cnn_lstm_train_loss = cnn_lstm_history.history['loss']
cnn_lstm_test_loss = cnn_lstm_history.history['val_loss']

epoch_count = range(1, len(cnn_lstm_train_loss)+1)

plt.plot(epoch_count, cnn_lstm_train_loss)
plt.plot(epoch_count, cnn_lstm_test_loss)
plt.title('loss history')
plt.legend(['train', 'validation'])
plt.xlabel('Epoch')
plt.xlabel('History')
plt.show()
#Normalized predictions:
cnn_lstm_train_pred = model_cnn_lstm.predict(X_train_series_sub)
cnn_lstm_valid_pred = model_cnn_lstm.predict(X_valid_series_sub)

print('Train rmse (avec normalisation):', np.sqrt(mean_squared_error(Y_train, cnn_lstm_train_pred)))
print('Validation rmse (avec normalisation):', np.sqrt(mean_squared_error(Y_valid, cnn_lstm_valid_pred)))
from sklearn.metrics import mean_absolute_error
print('Train mae (avec normalisation):', mean_absolute_error(Y_train, cnn_lstm_train_pred))
print('Validation mae (avec normalisation):', mean_absolute_error(Y_valid, cnn_lstm_valid_pred))
normalized_cnn_lstm_predictions = pd.DataFrame(Y_valid.values, columns=['Temperature'])
normalized_cnn_lstm_predictions.index = X_valid.index 
normalized_cnn_lstm_predictions['Predicted Temperature'] = cnn_lstm_valid_pred
normalized_cnn_lstm_predictions.head()
normalized_cnn_lstm_predictions.tail()
normalized_cnn_lstm_predictions.plot()
print(cnn_lstm_valid_pred)
print(normalized_cnn_lstm_predictions['Temperature'] )
y_val_cnn_lstm_dataset = np.zeros(shape=(len(cnn_lstm_valid_pred), 14) )
y_val_cnn_lstm_dataset[:,1] = normalized_cnn_lstm_predictions['Temperature']
y_val_inv_cnn_lstm = data_scaler.inverse_transform(y_val_cnn_lstm_dataset)[:,1]
print(y_val_inv_cnn_lstm)
pred_cnn_lstm_dataset = np.zeros(shape=(len(cnn_lstm_valid_pred), 14) )
pred_cnn_lstm_dataset[:,1] = cnn_lstm_valid_pred.reshape(17470)
y_pred_inv_cnn_lstm = data_scaler.inverse_transform(pred_cnn_lstm_dataset)[:,1]
cnn_lstm_predictions = pd.DataFrame(y_val.values, columns=['True Temperature'])
cnn_lstm_predictions.index = y_val.index 
cnn_lstm_predictions['Predicted Temperature'] = y_pred_inv_cnn_lstm
print(cnn_lstm_predictions)
cnn_lstm_predictions.to_csv('new-cnn-lstm-predictions.csv') 
from sklearn.metrics import mean_absolute_error
print('Validation mae (sans normalisation):', mean_absolute_error(cnn_lstm_predictions['True Temperature'], cnn_lstm_predictions['Predicted Temperature']))
cnn_lstm_predictions.plot()
import matplotlib.pyplot as plt 
cnn_lstm_true_temp = cnn_lstm_predictions['True Temperature']
time_stamp = cnn_lstm_predictions.index
plt.plot(time_stamp, cnn_lstm_true_temp)
plt.xlabel('Time')
plt.ylabel('True Temperature C°')
plt.show() 
import matplotlib.pyplot as plt 
cnn_lstm_pred_temp = cnn_lstm_predictions['Predicted Temperature']
time_stamp = cnn_lstm_predictions.index
plt.plot(time_stamp, cnn_lstm_pred_temp)
plt.xlabel('Time')
plt.xlabel('Predicted Temperature C°')
plt.show() 
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