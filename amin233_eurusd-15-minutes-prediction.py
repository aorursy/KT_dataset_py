PAIRS_LIST        = ["EURUSD", "USDJPY", "EURJPY"] #, "USDCHF", "EURCHF"] #, "AUDUSD" ]

PREDICTING_PAIR   = "EURUSD" 

PREDICTING_COLUMN = "close"

LOOK_BACK         = 30 # 20 * 15 six hour

SPLIT             = 0.95 # data split ration for training and testing
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
# install talib

%cd /kaggle/working

%rm -rf temp

!mkdir temp

%cd ./temp

!wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz

!tar -xzvf ta-lib-0.4.0-src.tar.gz

%cd ./ta-lib

!./configure --prefix=/usr

!make

!make install

!pip install Ta-Lib

%cd /kaggle/working

!rm -rf temp
# import libraries

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Activation, Input, LSTM, Dense, Flatten

from tensorflow.keras.models import load_model

from tensorflow.keras.callbacks import TensorBoard



from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error

from sklearn.decomposition import PCA, KernelPCA



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



# Used TA-Lib for creating additional features. More on this later.

from talib.abstract import *

from talib import MA_Type



import datetime
# loading data

data = {}



for pair_name in PAIRS_LIST:

    data[pair_name] = pd.read_csv("/kaggle/input/forex-top-currency-pairs-20002020/"+pair_name+"-2000-2020-15m.csv")



# normalize data shape and format

def norm_data_shape_format(df):

    orig_cols = ["DATE_TIME", "OPEN", "HIGH", "LOW", "CLOSE"]

    cols_name = ["timestamp", "open", "high", "low", "close"]

    df.rename(columns=dict(zip(orig_cols, cols_name)), inplace=True)

    df['timestamp'] = pd.to_datetime(df['timestamp'], infer_datetime_format=True)

    df.set_index("timestamp", inplace=True)

    df = df.reindex(columns=cols_name[1:])

    return df.astype(float)

    

for key in data:

    data[key] = norm_data_shape_format(data[key])
data["EURUSD"].head()
def extract_features(df):

    df['hour'] = df.index.hour

    df['day']  = df.index.weekday

    df['week'] = df.index.week

    # df['volume'] = pd.to_numeric(df['volume'])

    df['close']  = pd.to_numeric(df['close'])

    df['open']   = pd.to_numeric(df['open'])

    # df['momentum']   = df['volume'] * (df['open'] - df['close'])

    df['avg_price']  = (df['low'] + df['high'])/2

    df['range']      = df['high'] - df['low']

    df['ohlc_price'] = (df['low'] + df['high'] + df['open'] + df['close'])/4

    df['oc_diff']      = df['open'] - df['close']

    # df['spread_open']  = df['ask_open'] - df['bid_open']

    # df['spread_close'] = df['ask_close'] - df['bid_close']

    inputs = {

        'open'   : df['open'].values,

        'high'   : df['high'].values,

        'low'    : df['low'].values,

        'close'  : df['close'].values,

        'volume' : np.zeros(df['close'].shape[0]) # for sake of using TA lib

    }

    df['ema'] = MA(inputs, timeperiod=15, matype=MA_Type.T3)

    df['bear_power'] = df['low'] - df['ema']

    df['bull_power'] = df['high'] - df['ema']

    # Since computing EMA leave some of the rows empty, we want to remove them. (EMA is a lagging indicator)

    df.dropna(inplace=True)

    # Add 1D PCA vector as a feature as well. This helped increasing the accuracy by adding more variance to the feature set

    pca_input = df.drop('close', axis=1).copy()

    pca_features = pca_input.columns.tolist()

    pca = PCA(n_components=1)

    df['pca'] = pca.fit_transform(pca_input.values.astype('float32'))



columns_order = ["open", "high", "low", "close", "hour", "day", "week", "avg_price", "range", "ohlc_price", "oc_diff", "ema", "bear_power", "bull_power", "pca"]    
for key in data:

    extract_features(data[key])
data["EURUSD"].head()
# plt.plot(data['EURUSD'][2900:5150]["close"])

# plt.plot(data['EURUSD'][2900:5150]["ema"])

# plt.plot(data['EURUSD'][1900:2150]["bull_power"])

# plt.plot(data['EURUSD'][1900:2150]["bear_power"])

# plt.show()
# sort and rename column names

def sort_and_rename(df, suffix):

    cols_name = [c + suffix for c in columns_order]    

    df.rename(columns=dict(zip(columns_order,cols_name)), inplace=True)

    df = df.reindex(columns=cols_name, copy=False)

    return df

    

all_columns = [] # to save the correct order of data



for key in PAIRS_LIST:

    data[key] = sort_and_rename(data[key], "_" + key)

    all_columns += list(data[key].columns) # to save the correct order of data
# merge 

merged_data = pd.DataFrame(data[PAIRS_LIST[0]])

for key in PAIRS_LIST[1:]:

    merged_data = merged_data.merge(data[key], how="inner", left_index=True, right_index=True)

    

merged_data = merged_data.reindex(columns=all_columns)



# drop duplicate columns

for key in PAIRS_LIST[1:]:

    merged_data.drop(columns=[

        "hour_"+key,

        "day_" +key,

        "week_"+key,

    ],inplace=True)

    

merged_data.head()
# seeing correlation between columns

corr = merged_data.corr()

mask = np.zeros_like(corr, dtype=np.bool)

f, ax = plt.subplots(figsize=(15, 15))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, ax=ax)
merged_data.head()
def create_dataset(df, look_back=10):

    dataX, dataY = [], []

    for i in range(len(df)-look_back-1):

        a = df[i:(i+look_back)]

        dataX.append(a)

        dataY.append(df[i + look_back])

    return np.array(dataX), np.array(dataY)
# Scale reshape and group the data



target_column_name = PREDICTING_COLUMN + "_" + PREDICTING_PAIR



# Create scalers

scaler = MinMaxScaler()

scaled = pd.DataFrame(scaler.fit_transform(merged_data), columns=merged_data.columns)



x_scaler = MinMaxScaler(feature_range=(0, 1))

x_scaler = x_scaler.fit(merged_data.values.astype('float32'))

y_scaler = MinMaxScaler(feature_range=(0, 1))

y_scaler = y_scaler.fit(merged_data[target_column_name].values.astype('float32').reshape(-1,1))



# Create dataset

target_index = scaled.columns.tolist().index(target_column_name)

dataset = scaled.values.astype('float32')



X, y = create_dataset(dataset, look_back=LOOK_BACK)

y = y[:,target_index]



train_size = int(len(X) * SPLIT)

trainX = X[:train_size]

trainY = y[:train_size]

testX = X[train_size:]

testY = y[train_size:]
print("all data shape:", X.shape)

print("train data shape:", trainX.shape)

print("test data shape:", testX.shape)
def create_model():

    model = Sequential()

    model.add(LSTM(20, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))

    model.add(LSTM(20, return_sequences=True))

    model.add(LSTM(10, return_sequences=True))

    model.add(Dropout(0.2))

    model.add(LSTM(4, return_sequences=False))

    model.add(Dense(4, kernel_initializer='uniform', activation='relu'))

    model.add(Dense(1, kernel_initializer='uniform', activation='relu'))

    

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'mse'])

    print(model.summary())

    

    return model
model = create_model()
# Save the best weight during training.

from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint("weights.best.hdf5", monitor='val_mse', verbose=1, save_best_only=True, mode='min')



# Monitor the trianing progress via TensorBoard

# log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# tensorboard = TensorBoard(log_dir=log_dir)



# Callbacks

callbacks_list = [checkpoint] # , tensorboard]



# Fit

history = model.fit(trainX, trainY, epochs=200, batch_size=512, verbose=1, callbacks=callbacks_list, validation_split=0.1)
def visualize_history():

    epoch = len(history.history['loss'])

    for k in list(history.history.keys()):

        if 'val' not in k:

            plt.figure(figsize=(40,10))

            plt.plot(history.history[k])

            plt.plot(history.history['val_' + k])

            plt.title(k)

            plt.ylabel(k)

            plt.xlabel('epoch')

            plt.legend(['train', 'test'], loc='upper left')

            plt.show()
visualize_history()
# To improve the weights towards the global optimal, I retrained the model with LearningRateScheduler added



from keras.callbacks import LearningRateScheduler

import keras.backend as K

def scheduler(epoch):

    if epoch%10==0 and epoch!=0:

        lr = K.get_value(model.optimizer.lr)

        K.set_value(model.optimizer.lr, lr*.9)

        print("lr changed to {}".format(lr*.9))

    return K.get_value(model.optimizer.lr)

lr_decay = LearningRateScheduler(scheduler)

callbacks_list = [checkpoint, lr_decay] # , tensorboard]

history = model.fit(trainX, trainY, epochs=3, batch_size=1024, callbacks=callbacks_list, validation_split=0.1)
visualize_history()
model.load_weights("weights.best.hdf5") # load best validation 
pred = model.predict(testX)
from sklearn.metrics import mean_absolute_error 



predictions = pd.DataFrame()

predictions['predicted'] = pd.Series(np.reshape(pred, (pred.shape[0])))

predictions['actual'] = testY

predictions = predictions.astype(float)



predictions.plot(figsize=(20,10))

plt.show()



predictions['diff'] = predictions['predicted'] - predictions['actual']

plt.figure(figsize=(10,10))

sns.distplot(predictions['diff']);

plt.title('Distribution of differences between actual and prediction')

plt.show()



print("MSE : ", mean_squared_error(predictions['predicted'].values, predictions['actual'].values))

print("MAE : ", mean_absolute_error(predictions['predicted'].values, predictions['actual'].values))

predictions['diff'].describe()
pred = model.predict(testX)
pred = y_scaler.inverse_transform(pred)

close = y_scaler.inverse_transform(np.reshape(testY, (testY.shape[0], 1)))



predictions = pd.DataFrame()

predictions['predicted'] = pd.Series(np.reshape(pred, (pred.shape[0])))

predictions['close'] = pd.Series(np.reshape(close, (close.shape[0])))
p = merged_data[-pred.shape[0]:].copy()

predictions.index = p.index

predictions = predictions.astype(float)

predictions = predictions.merge(p[['low_'+PREDICTING_PAIR, 'high_'+PREDICTING_PAIR]], right_index=True, left_index=True)
zoom = 200



ax = predictions[:zoom].plot(y='close', c='red', figsize=(40,10))

ax = predictions[:zoom].plot(y='predicted', c='blue', figsize=(40,10), ax=ax)

index = [str(item) for item in predictions[:zoom].index]

plt.fill_between(x=index, y1='low_'+PREDICTING_PAIR, y2='high_'+PREDICTING_PAIR, data=p[:zoom], alpha=0.4)

plt.title('Prediction vs Actual (low and high as blue region)')

plt.show()
predictions['diff'] = predictions['predicted'] - predictions['close']

plt.figure(figsize=(10,10))

sns.distplot(predictions['diff']);

plt.title('Distribution of differences between actual and prediction ')

plt.show()
g = sns.jointplot("diff", "predicted", data=predictions, kind="kde", space=0)

plt.title('Distributtion of error and price')

plt.show()
print("MSE : ", mean_squared_error(predictions['predicted'].values, predictions['close'].values))

print("MAE : ", mean_absolute_error(predictions['predicted'].values, predictions['close'].values))

predictions['diff'].describe()