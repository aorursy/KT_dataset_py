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


import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
tf.random.set_seed(47)
df = pd.read_pickle('/kaggle/input/hec-simulation-data/HEC1Front_ETA2.35_mu200_BC-1.pkl')
df.head()
fig, ax = plt.subplots(1, 3, figsize=(15,4))
ax[0].plot(df['signal'].values[3000:3500])
ax[0].set_xlabel('Bunch Crossing [25 ns]')
ax[0].set_ylabel('Energy [GeV]')
ax[0].set_title('Signal')
ax[1].plot(df['pileup'].values[3000:3500])
ax[1].set_xlabel('Bunch Crossing [25 ns]')
ax[1].set_ylabel('Energy [GeV]')
ax[1].set_title('Pileup')
ax[2].plot(df['ADC'].values[3000:3500])
ax[2].set_xlabel('Bunch Crossing [25 ns]')
ax[2].set_ylabel('Energy [GeV]')
ax[2].set_title('LAr Electronics')
plt.show()
def get_ML_data(df):
    df = df[['signal', 'pileup', 'ADC', 'OFC', 'mu']]
    start = 3000 
    df = df[start::]
    
    trigger_points = df['signal'].values>0.2

    # Signal as Signal and Pileup
    df['signal'] = df['signal'] + df['pileup']

    # Normalize Data
    ADC_mean = np.mean(df['ADC'])
    ADC_std = np.std(df['ADC'])
    df['ADC_norm'] = (df['ADC'] - ADC_mean)/ADC_std
    
    extra_bcs = 3
    X = np.expand_dims([np.concatenate([df['ADC'].values[1:], np.array([0])])] , -1)
    y = np.expand_dims(np.concatenate([np.zeros(extra_bcs), df['signal'].values[0:-extra_bcs]]), 0)
    w = 1*np.concatenate([np.zeros(extra_bcs), trigger_points[0:-extra_bcs]])

    X = X.reshape(9999,3000,1)
    y = y.reshape(9999,3000, 1)
    w = w.reshape(9999, 3000)
    
    data_split = 8000
    
    X_train = X[:data_split]
    X_valid = X[data_split:]
    y_train = y[:data_split]
    w_train = w[:data_split]
    y_valid= y[data_split:]
    w_valid = w[data_split:]

    return (X_train, y_train, w_train), (X_valid, y_valid, w_valid)
data = (X_train, y_train, w_train), (X_valid, y_valid, w_valid) = get_ML_data(df)
def scheduler(epoch):
    if epoch < 20:
        return 1e-2
    elif epoch < 150:
        return 1e-3
    else:
        return 1e-4

# Constructs model and returns an untrained model
def create_model():
    input_layer = keras.layers.Input(shape=[None, 1])
    # Dilation rates of the network
    for i, rate in enumerate([1, 2, 2]):
        if i==0:
            layer_curr = input_layer
        layer_curr = keras.layers.Conv1D(filters=3, kernel_size=2, padding="causal",
                                      activation="relu", dilation_rate=rate)(layer_curr)
        
    layer_curr_signal = keras.layers.Conv1D(filters=1, kernel_size=3, padding="causal", activation='relu')(layer_curr)

    return keras.Model(inputs =[input_layer], outputs=[layer_curr_signal])

# Trains the model on the data provided
def train_model(data, model):
    (X_train, y_train, w_train), (X_valid, y_valid, w_valid) = data
    
    opt = keras.optimizers.Nadam(learning_rate=0.001)
    model.compile(loss="mse", optimizer=opt, sample_weight_mode='temporal')
    history = model.fit(X_train, y_train, sample_weight=w_train,
                        epochs=200, batch_size=16,
                        callbacks=[tf.keras.callbacks.LearningRateScheduler(scheduler)])
model = create_model()
history = train_model(data, model)