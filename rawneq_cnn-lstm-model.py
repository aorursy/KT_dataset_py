import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import optimizers
from keras.utils import plot_model
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten, Dropout, Bidirectional
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
from sklearn.preprocessing import MinMaxScaler

%matplotlib inline
warnings.filterwarnings("ignore")
init_notebook_mode(connected=True)

# Set seeds to make the experiment more reproducible.
import tensorflow as tf
from numpy.random import seed
tf.random.set_seed(1)
seed(1)
data = pd.read_csv('../input/climate-hour/climate_hour.csv', parse_dates=['Date Time'],index_col = 0, header=0) 
data = data.sort_values(['Date Time'])
data.head() 
temp_data = data['T (degC)']                               
temp_data = pd.DataFrame({'Date Time': data.index, 'T (degC)':temp_data.values})
temp_data = temp_data.set_index(['Date Time'])
temp_data.head()
temp_scaler = MinMaxScaler()
temp_scaler.fit(temp_data) 
normalized_temp = temp_scaler.transform(temp_data) 
normalized_temp = pd.DataFrame(normalized_temp, columns=['T (degC)'])
normalized_temp.index = temp_data.index
normalized_temp.head()
# Normalized data:
scaler = MinMaxScaler()
scaler.fit(data) 
normalized_df = scaler.transform(data) 
normalized_df = pd.DataFrame(normalized_df, columns=['p (mbar)','T (degC)','Tpot (K)','Tdew (degC)',
                                                         'rh (%)','VPmax (mbar)','VPact (mbar)',
                                                         'VPdef (mbar)','sh (g/kg)','H2OC (mmol/mol)',
                                                         'rho (g/m**3)','wv (m/s)', 'max. wv (m/s)','wd (deg)'])
normalized_df.index = data.index
normalized_df.head()
#old
def series_to_supervised(data, window=1, lag=1, dropnan=True):
    cols, names = list(), list()
    # Input sequence (t-n, ... t-1)
    for i in range(window, 0, -1):
        cols.append(data.shift(i))
        names += [('%s(t-%d)' % (col, i)) for col in data.columns]
    # Current timestep (t=0)
    cols.append(data)
    names += [('%s(t)' % (col)) for col in data.columns]
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
# -> 2009-01-02 01:00:00 
labels_col = 'T (degC)(t)'
labels = series[labels_col]
series = series.drop(['p (mbar)(t)', 'T (degC)(t)', 'Tpot (K)(t)',
                      'Tdew (degC)(t)' ,'rh (%)(t)', 'VPmax (mbar)(t)',
                      'VPact (mbar)(t)', 'VPdef (mbar)(t)','sh (g/kg)(t)', 
                      'H2OC (mmol/mol)(t)', 'rho (g/m**3)(t)', 'wv (m/s)(t)',
                      'max. wv (m/s)(t)', 'wd (deg)(t)'], axis=1)

X_train = series['2009-01-02 01:00:00':'01.01.2015 00:00:00']
X_valid = series['01.01.2015 00:00:00':'2017-01-01 00:00:00'] 
Y_train = labels['2009-01-02 01:00:00':'01.01.2015 00:00:00']
Y_valid = labels['01.01.2015 00:00:00':'2017-01-01 00:00:00']
print('Train set shape', X_train.shape)
print('Validation set shape', X_valid.shape)
import time
name = "model-mlp{}".format(int(time.time()))
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.layers import Dense,RepeatVector, LSTM, Dropout
from tensorflow.keras.layers import Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.layers import Bidirectional, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
X_train_series = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
X_valid_series = X_valid.values.reshape((X_valid.shape[0], X_valid.shape[1], 1))
print('Train set shape', X_train_series.shape)
print('Validation set shape', X_valid_series.shape)
!pip install h5py
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping
early_stop = EarlyStopping(monitor = "loss", mode = "min", patience = 7)
checkpoint = ModelCheckpoint('best_model.h5'.format(int(time.time())), monitor='loss', mode='min', save_best_only=True, verbose=1)
# detect and init the TPU
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)

# instantiate a distribution strategy
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

# instantiating the model in the strategy scope creates the model on the TPU
with tpu_strategy.scope():
    model = Sequential()
    model.add(Conv1D(filters=256, kernel_size=2, activation='relu', input_shape=(X_train_series.shape[1], X_train_series.shape[2])))
    #model.add(Conv1D(filters=128, kernel_size=2, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(RepeatVector(30))
    #model.add(LSTM(units=25, return_sequences=True, activation='relu'))
    #model.add(Dropout(0.2))
    #model.add(LSTM(units=25, return_sequences=True, activation='relu'))
    #model.add(Dropout(0.2))
    #model.add(LSTM(units=25, return_sequences=True, activation='relu'))
    model.add(LSTM(units=50, return_sequences=True, activation='relu'))
    model.add(Bidirectional(LSTM(128, activation='relu')))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam', metrics=['mse'])
plot_model(model, to_file='model.png')
history = model.fit(X_train_series, Y_train, validation_data=(X_valid_series, Y_valid), epochs=10, callbacks=[early_stop, checkpoint], verbose=1)
from tensorflow.keras.models import load_model
saved_model = load_model('best_model.h5')
#model.save('best_model.h5')
'''
model_csv = saved_model.to_csv()
with open("saved_model.csv", "w") as csv_file:
    csv_file.write(model_csv)
'''    
model_json = saved_model.to_json()
with open("saved_model.json", "w") as json_file:
    json_file.write(model_json)
import matplotlib.pyplot as plt 
train_loss = history.history['loss']
test_loss = history.history['val_loss']

epoch_count = range(1, len(train_loss)+1)

plt.plot(epoch_count, train_loss)
plt.plot(epoch_count, test_loss)
plt.title('loss history')
plt.legend(['train', 'validation'])
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.show()
Y_train
#Normalized predictions:
train_pred = model.predict(X_train_series)
valid_pred = model.predict(X_valid_series)

print('Train rmse (avec normalisation):', np.sqrt(mean_squared_error(Y_train, train_pred[:-1])))
print('Validation rmse (avec normalisation):', np.sqrt(mean_squared_error(Y_valid, valid_pred[:-1])))
from sklearn.metrics import mean_absolute_error
print('Train mae (avec normalisation):', mean_absolute_error(Y_train, train_pred[:-1]))
print('Validation mae (avec normalisation):', mean_absolute_error(Y_valid, valid_pred[:-1]))
normalized_lstm_predictions = pd.DataFrame(Y_valid.values, columns=['Temperature'])
normalized_lstm_predictions.index = X_valid.index 
normalized_lstm_predictions['Predicted Temperature'] = valid_pred[:-1]
normalized_lstm_predictions.head()
normalized_lstm_predictions.plot()
Y_valid_inv = temp_scaler.inverse_transform(Y_valid.values.reshape(-1, 1))
pred_inv = temp_scaler.inverse_transform(valid_pred)
model_predictions = pd.DataFrame(Y_valid_inv, columns=['Temperature'])
model_predictions.index = X_valid.index 
model_predictions['Predicted Temperature'] = pred_inv[:-1]
model_predictions.head()
import time
model_predictions.to_csv('model-predictions{}.csv'.format(int(time.time())))
from sklearn.metrics import mean_absolute_error
print('Validation mae (sans normalisation):', mean_absolute_error(Y_valid_inv, pred_inv[:-1]))
def series_to_supervised(data, window=1, lag=1, dropnan=True, single=True):
    cols, names = list(), list()
    # Input sequence (t-n, ... t-1)
    for i in range(window, 0, -1):
        cols.append(data.shift(i))
        names += [('%s(t-%d)' % (col, i)) for col in data.columns]
    # Current timestep (t=0)
    cols.append(data)
    names += [('%s(t)' % (col)) for col in data.columns]
    
    #Single step & Multi step
    if single:
        cols.append(data.shift(-lag))
        names += [('%s(t+%d)' % (col, lag)) for col in data.columns]
    else:
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
from sklearn.preprocessing import MinMaxScaler
data_scaler = MinMaxScaler()
data_scaler.fit(data) 
normalized_data = data_scaler.transform(data) 

normalized_df = pd.DataFrame(normalized_data, columns=['p (mbar)','T (degC)','Tpot (K)','Tdew (degC)',
                                                         'rh (%)','VPmax (mbar)','VPact (mbar)',
                                                         'VPdef (mbar)','sh (g/kg)','H2OC (mmol/mol)',
                                                         'rho (g/m**3)','wv (m/s)', 'max. wv (m/s)','wd (deg)'])
normalized_df = normalized_df.set_index(data.index)
normalized_df.head()
window = 24
lag = 24
single_series = series_to_supervised(normalized_df, window=window, lag=lag)
single_series.head()
labels_col = 'T (degC)(t+24)'
labels = single_series[labels_col]
single_series = single_series.drop(['p (mbar)(t+24)','T (degC)(t+24)','Tpot (K)(t+24)','Tdew (degC)(t+24)','rh (%)(t+24)','VPmax (mbar)(t+24)','VPact (mbar)(t+24)','VPdef (mbar)(t+24)','sh (g/kg)(t+24)','H2OC (mmol/mol)(t+24)','rho (g/m**3)(t+24)','wv (m/s)(t+24)','max. wv (m/s)(t+24)','wd (deg)(t+24)'], axis=1)
X_train = single_series['2009-01-02 01:00:00':'01.01.2015 00:00:00']
X_valid = single_series['01.01.2015 00:00:00':'2017-01-01 00:00:00'] 
Y_train = labels['2009-01-02 01:00:00':'01.01.2015 00:00:00']
Y_valid = labels['01.01.2015 00:00:00':'2017-01-01 00:00:00']
print('Train set shape', X_train.shape)
print('Validation set shape', X_valid.shape)
print('Train set shape', Y_train.shape)
print('Validation set shape', Y_valid.shape)
X_train_reshaped = X_train.values.reshape(X_train.shape[0],25,14)
X_valid_reshaped = X_valid.values.reshape(X_valid.shape[0],25,14)
Y_train_reshaped = Y_train.values.reshape(Y_train.shape[0],)
Y_valid_reshaped = Y_valid.values.reshape(Y_valid.shape[0],)
print('Train set shape',X_train_reshaped.shape,Y_train_reshaped.shape)
print('Validation set shape',X_valid_reshaped.shape,Y_valid_reshaped.shape)
print(X_train_reshaped.shape[1])
print(X_train_reshaped.shape[2])
single_model = Sequential()
single_model.add(Conv1D(filters=256, kernel_size=2, activation='relu', input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
    #model.add(Conv1D(filters=128, kernel_size=2, activation='relu'))
single_model.add(MaxPooling1D(pool_size=2))
single_model.add(Flatten())
single_model.add(RepeatVector(30))
    #model.add(LSTM(units=25, return_sequences=True, activation='relu'))
    #model.add(Dropout(0.2))
    #model.add(LSTM(units=25, return_sequences=True, activation='relu'))
    #model.add(Dropout(0.2))
    #model.add(LSTM(units=25, return_sequences=True, activation='relu'))
single_model.add(LSTM(units=50, return_sequences=True, activation='relu'))
single_model.add(Bidirectional(LSTM(128, activation='relu')))
single_model.add(Dense(50, activation='relu'))
single_model.add(Dense(1))
single_model.compile(loss='mae', optimizer='adam', metrics=['mse'])
single_history = single_model.fit(X_train_reshaped, Y_train_reshaped, validation_data=(X_valid_reshaped, Y_valid_reshaped), epochs=10, verbose=1)
#Ploting history:
import matplotlib.pyplot as plt 
single_loss = single_history.history['loss']
single_val_loss = single_history.history['val_loss']

epoch_count = range(1, len(single_loss)+1)

plt.plot(epoch_count, single_loss)
plt.plot(epoch_count, single_val_loss)
plt.title('loss history')
plt.legend(['train', 'validation'])
plt.xlabel('Epoch')
plt.ylabel('Loss value')
plt.show()
#Normalized predictions:
from sklearn.metrics import mean_absolute_error

single_train_pred = single_model.predict(X_train_reshaped)
single_valid_pred = single_model.predict(X_valid_reshaped)

print('Train MAE (avec normalisation):', mean_absolute_error(Y_train_reshaped, single_train_pred))
print('Validation MAE (avec normalisation):', mean_absolute_error(Y_valid_reshaped, single_valid_pred))
#Saving history in a csv file :
import time
single_step_hist_df = pd.DataFrame(single_history.history) 
single_step_csv_file = 'single-step-history-{}.csv'.format(int(time.time()))
with open(single_step_csv_file, mode='w') as file:
    single_step_hist_df.to_csv(file)
single_normalized_predictions = pd.DataFrame(Y_valid.values, columns=['Temperature'])
single_normalized_predictions.index = X_valid.index 
single_normalized_predictions['Predicted Temperature'] = single_valid_pred
single_normalized_predictions.head()
single_normalized_predictions.plot()
temp_data = data['T (degC)']                               
temp_data = pd.DataFrame({'Date Time': data.index, 'T (degC)':temp_data.values})
temp_data = temp_data.set_index(['Date Time'])
temp_data.head()
from sklearn.preprocessing import MinMaxScaler
temp_scaler = MinMaxScaler()
temp_scaler.fit(temp_data) 
normalized_temp = temp_scaler.transform(temp_data) 
from sklearn.metrics import mean_absolute_error

y_val = temp_data['01.01.2015 00:00:00':'2016-12-31 00:00:00']
single_pred_inv = temp_scaler.inverse_transform(single_valid_pred)

print('Validation mae (sans normalisation):', mean_absolute_error(y_val, single_pred_inv))
single_predictions = pd.DataFrame(y_val.values, columns=['True Temperature'])
single_predictions.index = y_val.index 
single_predictions['Predicted Temperature'] = single_pred_inv
print(single_predictions)
single_predictions.plot()
single_predictions.to_csv('single-step-normal-predictions{}.csv'.format(int(time.time())))
window = 72
lag = 24
multi_series = series_to_supervised(normalized_df, window=window, lag=lag, single=False)
multi_series.head()
temp_cols = [('T (degC)(t+%d)' % (i))for i in range(1, lag+1)]
multi_temp = multi_series[temp_cols]
columns = [('%s(t+%d)' % (col,lg)) for col in data.columns for lg in range(1,lag+1)]

multi_series = multi_series.drop(columns, axis=1)
multi_series.head()
X_train_multi = multi_series['2009-01-02 01:00:00':'01.01.2015 00:00:00']
X_valid_multi = multi_series['01.01.2015 00:00:00':'2017-01-01 00:00:00'] 
Y_train_multi = multi_temp['2009-01-02 01:00:00':'01.01.2015 00:00:00']
Y_valid_multi = multi_temp['01.01.2015 00:00:00':'2017-01-01 00:00:00']
print('Train set shape', X_train_multi.shape,Y_train_multi.shape)
print('Validation set shape', X_valid_multi.shape, Y_valid_multi.shape)
x_train_multi = X_train_multi.values.reshape(X_train_multi.shape[0],73,14)
x_valid_multi = X_valid_multi.values.reshape(X_valid_multi.shape[0],73,14)
y_train_multi = Y_train_multi.values.reshape(Y_train_multi.shape[0],24)
y_valid_multi = Y_valid_multi.values.reshape(Y_valid_multi.shape[0],24)

print('Train set shape', x_train_multi.shape,y_train_multi.shape)
print('Validation set shape', x_valid_multi.shape, y_valid_multi.shape)
# (x_train_multi.shape[1], x_train_multi.shape[2])
multi_model = Sequential()
multi_model.add(Conv1D(filters=256, kernel_size=2, activation='relu', input_shape=(x_train_multi.shape[1], x_train_multi.shape[2])))
    #model.add(Conv1D(filters=128, kernel_size=2, activation='relu'))
multi_model.add(MaxPooling1D(pool_size=2))
multi_model.add(Flatten())
multi_model.add(RepeatVector(30))
    #model.add(LSTM(units=25, return_sequences=True, activation='relu'))
    #model.add(Dropout(0.2))
    #model.add(LSTM(units=25, return_sequences=True, activation='relu'))
    #model.add(Dropout(0.2))
    #model.add(LSTM(units=25, return_sequences=True, activation='relu'))
multi_model.add(LSTM(units=50, return_sequences=True, activation='relu'))
multi_model.add(Bidirectional(LSTM(128, activation='relu')))
multi_model.add(Dense(50, activation='relu'))
multi_model.add(Dense(24))
multi_model.compile(loss='mae', optimizer='adam', metrics=['mse'])
multi_step_history = multi_model.fit(x_train_multi, y_train_multi, validation_data=(x_valid_multi, y_valid_multi), epochs=10, verbose=1)
#Ploting history:
import matplotlib.pyplot as plt 
multi_step_loss = multi_step_history.history['loss']
multi_step_val_loss = multi_step_history.history['val_loss']

epoch_count = range(1, len(multi_step_loss)+1)

plt.plot(epoch_count, multi_step_loss)
plt.plot(epoch_count, multi_step_val_loss)
plt.title('Multi step model - loss history')
plt.legend(['train', 'validation'])
plt.xlabel('Epoch')
plt.ylabel('Loss value')
plt.show()
#Ploting history:
import matplotlib.pyplot as plt 
multi_step_loss = multi_step_history.history['mse']
multi_step_val_loss = multi_step_history.history['val_mse']

epoch_count = range(1, len(multi_step_loss)+1)

plt.plot(epoch_count, multi_step_loss)
plt.plot(epoch_count, multi_step_val_loss)
plt.title('Multi step model - loss history')
plt.legend(['train', 'validation'])
plt.xlabel('Epoch')
plt.ylabel('Loss value')
plt.show()
# save the model to disk
import pickle
filename = 'multi_step_model.sav'
pickle.dump(multi_model, open(filename, 'wb'))
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
#Normalized predictions:
from sklearn.metrics import mean_absolute_error

multi_train_pred = multi_model.predict(x_train_multi)
multi_valid_pred = multi_model.predict(x_valid_multi)

print('Train MAE (avec normalisation):', mean_absolute_error(y_train_multi, multi_train_pred))
print('Validation MAE (avec normalisation):', mean_absolute_error(y_valid_multi, multi_valid_pred))
true_temp = pd.DataFrame(Y_valid_multi.values, columns=['%s h'%i for i in range(1,25)], index=X_valid_multi.index)
pred_temp = pd.DataFrame(multi_valid_pred, columns=['%s h'%i for i in range(1,25)], index=X_valid_multi.index)

multi_normalized_predictions = pd.concat([true_temp, pred_temp], axis=1)
print(multi_normalized_predictions)

multi_normalized_predictions.head()
temp_data = data['T (degC)']                               
temp_data = pd.DataFrame({'Date Time': data.index, 'T (degC)':temp_data.values})
temp_data = temp_data.set_index(['Date Time'])
temp_data.head()
from sklearn.preprocessing import MinMaxScaler
temp_scaler = MinMaxScaler()
temp_scaler.fit(temp_data) 
normalized_temp = temp_scaler.transform(temp_data)
y_val = temp_data['01.01.2015 00:00:00':'2016-12-31 00:00:00']
Y_valid_multi_inv = temp_scaler.inverse_transform(Y_valid_multi.values)
multi_pred_inv = temp_scaler.inverse_transform(multi_valid_pred)
tr = pd.DataFrame(Y_valid_multi_inv, columns=['True %s h'%i for i in range(1,25)], index=X_valid_multi.index)
pr = pd.DataFrame(multi_pred_inv, columns=['Pred %s h'%i for i in range(1,25)], index=X_valid_multi.index)

multi_predictions = pd.concat([tr, pr], axis=1) 
multi_predictions.head()
import time
multi_predictions.to_csv('multi-step-normal-predictions{}.csv'.format(int(time.time())))
from sklearn.metrics import mean_absolute_error
print('Validation mae (sans normalisation):', mean_absolute_error(Y_valid_multi_inv, multi_pred_inv))
multi_predictions.plot(figsize=(20,15))
#multi_predictions.plot(subplot=True, figsize=(20,15))
couples = []
for i in range(1,24):
    tr = str('True '+str(i)+' h')
    pr = str('Pred '+str(i)+' h')
    #c = [tr,pr]
    couples.append([tr,pr])
    #multi_predictions[[tr , pr ]].plot(figsize=(20,15))

true_pred_data = []
for c in couples:
    true_pred_data.append(multi_predictions[c])#.plot(figsize=(15,11))
fig, axs = plt.subplots(23, figsize=(15,20))
fig.suptitle('True vs Pred')
i=0
for c in couples:
    #true_pred_data.append(multi_predictions[c])#.plot(figsize=(15,11))
    axs[i].plot(multi_predictions[c])
    axs[i].set_title(c)
    i=i+1
for c in couples:
    multi_predictions[c].plot()