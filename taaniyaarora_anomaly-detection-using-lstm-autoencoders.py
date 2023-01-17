!nvidia-smi
!pip install tensorflow-gpu
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, MinMaxScaler

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.layers import LSTM, Dense, RepeatVector, Dropout, TimeDistributed

import seaborn as sns



import warnings

warnings.simplefilter('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv("/kaggle/input/sp500-daily-19862018/spx.csv", parse_dates=['date'], index_col='date')
df.plot(figsize=(14,8))

plt.show()
df.info()
df.describe()
### Using 95% as training data



# We'll look back 30 days of historical data to learn past trend. 

# Setting shuffle to False to retain the time series

TIMESTEPS = 30            



train_data, test_data = train_test_split(df, train_size=0.95, shuffle=False)

train_data.sort_index(inplace=True)

test_data.sort_index(inplace=True)

train_data.shape, test_data.shape
train_data
train_data.info()
test_data.info()
def getScaledData(method='standard', train_df=None, test_df=None, feature_col='feature'):

    if method == 'standard':

        scaler = StandardScaler()

    else:

        scaler = MinMaxScaler()

    scaler = scaler.fit(train_df[[feature_col]])

    train_df['scaled_'+feature_col] = scaler.transform(train_df[[feature_col]])

    test_df['scaled_'+feature_col] = scaler.transform(test_df[[feature_col]])

    return train_df, test_df, scaler

    

def createDataset(df, lookback=30, feature_col=None):

    data_x, data_y = [], []

    for i in range(lookback, len(df)):

        data_x.append(df.iloc[i-lookback:i][[feature_col]].values)

        data_y.append(df.iloc[i][feature_col])

    data_x = np.array(data_x)

    data_y = np.array(data_y)

    return data_x, data_y
train_df, test_df, scaler = getScaledData('standard', train_data, test_data, 'close')

train_df.shape, test_df.shape
train_df['scaled_close'].plot(figsize=(14,8))

plt.show()
train_x, train_y = createDataset(train_df, TIMESTEPS, 'scaled_close')

test_x, test_y = createDataset(test_df, TIMESTEPS, 'scaled_close')
train_x.shape, train_y.shape, test_x.shape, test_y.shape


LSTM_units = 64

model = keras.Sequential()

model.add(LSTM(LSTM_units, input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences=False,name='encoder_lstm'

              ))

model.add(Dropout(0.2, name='encoder_dropout'))

model.add(RepeatVector(train_x.shape[1], name='decoder_repeater'))

model.add(LSTM(LSTM_units, return_sequences=True, name='decoder_lstm'))

model.add(Dropout(rate=0.2, name='decoder_dropout'))

model.add(TimeDistributed(Dense(train_x.shape[2],name='decoder_dense_output')))



model.compile(loss='mae', optimizer='adam')
model.summary()
%time history = model.fit(train_x, train_x, epochs=10, batch_size=32, validation_split=0.1, shuffle=False)
plt.plot(history.history['loss'], label='training_loss')

plt.plot(history.history['val_loss'], label='validation_loss')

plt.legend()

plt.show()
reconstructed = model.predict(train_x)

reconstructed.shape, train_x.shape
# Reconstruction error - MAE for each sample



mae_loss = np.mean(np.abs(reconstructed - train_x), axis=1)

mae_loss.shape
sns.distplot(mae_loss[:,0])

plt.show()
THRESHOLD = 0.65
test_reconstruction = model.predict(test_x)

test_reconstruction.shape
# MAE for reconstruction on test data

test_mae_loss = np.mean(np.abs(test_x - test_reconstruction), axis=1)

test_mae_loss.shape
test_df.info()
# Setting index after N timesteps from past in test_df

anomaly_results_df = test_df[TIMESTEPS:][['close', 'scaled_close']].copy()

anomaly_results_df.index = test_df[TIMESTEPS:].index



# Including reconstructed predictions

anomaly_results_df['deviation'] = test_mae_loss

anomaly_results_df['threshold'] = THRESHOLD

anomaly_results_df['anomaly'] = anomaly_results_df['deviation'].apply(lambda dev: 1 if dev > THRESHOLD else 0)





anomalies = anomaly_results_df[anomaly_results_df['anomaly'] == 1]

anomalies.shape
anomaly_results_df['anomaly'].plot(kind='hist')

plt.show()
anomaly_results_df[['deviation', 'threshold']].plot(figsize=(14, 6))

plt.show()
anomaly_results_df[['close']].plot(figsize=(14, 6))

sns.scatterplot(anomalies.index, anomalies['close'],label='anomaly',color='red')

plt.show()