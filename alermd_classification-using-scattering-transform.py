import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath('../input/'))
from utils import ESC50

import scipy
from scipy.io import wavfile
from scipy import signal

import tensorflow as tf
from tensorflow.keras import layers
 
from kymatio.keras import Scattering1D
shared_params = {'csv_path': 'esc50.csv',
                 'wav_dir': 'audio/audio',
                 'dest_dir': 'audio/audio/16000',
                 'audio_rate': 16000,
                 'only_ESC10': True,
                 'pad': 0,
                 'normalize': True}

train_gen = ESC50(folds=[1,2,3,4],
                  randomize=False,
                  strongAugment=True,
                  random_crop=True,
                  inputLength=2,
                  mix=False,
                  **shared_params).batch_gen(100)

test_gen = ESC50(folds=[5],
                  randomize=True,
                  strongAugment=True,
                  random_crop=True,
                  inputLength=2,
                  mix=False,
                  **shared_params).batch_gen(100)
X_tr, Y_tr = next(train_gen)

X_train = []
for x in X_tr:
    X_train.append(x[:, 0])
    
X_train = np.array(X_train)

Y_train = []
for y in Y_tr:
    Y_train.append(np.where(y==1)[0][0])
    
Y_train = np.array(Y_train)
X_t, Y_ts = next(test_gen)

X_test = []
for x in X_t:
    X_test.append(x[:, 0])
    
X_test = np.array(X_test)

Y_test = []
for y in Y_ts:
    Y_test.append(np.where(y==1)[0][0])
    
Y_test = np.array(Y_test)
X_train.shape
X_test.shape
# maximum scale 2**J of the scattering transform
J = 8

# the number of wavelets per octave.
Q = 12

# define a small constant to add to the scattering coefficients before computing the logarithm.
# this prevents very large values when the scattering coefficients are very close to zero.
log_eps = 1e-6
x_in = layers.Input(shape=(X_train.shape[1]))
x = Scattering1D(J, Q=Q)(x_in)
x = layers.Lambda(lambda x: x[..., 1:, :])(x)
x = layers.Lambda(lambda x: tf.math.log(tf.abs(x) + log_eps))(x)
x = layers.GlobalAveragePooling1D(data_format='channels_first')(x)
x = layers.BatchNormalization(axis=1)(x)
x_out = layers.Dense(10, activation='softmax')(x)
model = tf.keras.models.Model(x_in, x_out)
model.summary()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
cnnhistory = model.fit(
                    X_train,
                    Y_train,
                    epochs=50,
                    batch_size=10,
                    validation_split=0.2)
plt.plot(cnnhistory.history['loss'])
plt.plot(cnnhistory.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
model.evaluate(X_train, Y_test, verbose=2)