import tensorflow as tf
import keras.backend as K

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))

import keras

from keras.optimizers import Adam
from keras.layers import Input, Dropout, Dense, LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
# Read train and test files
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
X_train_orig = train_df.drop(["ID", "target"], axis=1)
X_test_orig = test_df.drop(["ID"], axis=1)

# Apply log transform to the features (it helps network a lot)
X_train = np.log1p(X_train_orig)
X_test = np.log1p(X_test_orig)

X_all = pd.concat((X_test, X_train), axis=0).replace(0,  np.nan) # exclude zeros from mean/std calculation

# Scale features
X_train = (X_train - X_all.mean()) / X_all.std()
X_test = (X_test - X_all.mean()) / X_all.std()

# Apply log transform to target variable
y_train = np.log1p(train_df["target"].values)
dev_X, val_X, dev_y, val_y = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)
input_layer = Input(shape=(X_train.shape[1],))
x = input_layer
x = Dense(1024, activation='linear')(x)
x = LeakyReLU()(x)
#x = Dropout(0.5)(x)
x = Dense(512, activation='linear')(x)
x = LeakyReLU()(x)
#x = Dropout(0.1)(x)
x = Dense(64, activation='relu')(x)
x = Dense(1, activation='linear')(x)
keras_nn = Model(inputs=input_layer, outputs=x, name='nn_zero')
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

keras_nn.compile(optimizer=Adam(lr=0.0001), loss=root_mean_squared_error, metrics=[root_mean_squared_error])
batch_size = 16
epochs = 200

lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.000001)
es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00005, patience=3, verbose=0, mode='auto')

history = keras_nn.fit(dev_X, dev_y, 
          epochs=epochs,
          batch_size=batch_size,
          callbacks=[lr_scheduler, es],
          validation_data=(val_X, val_y))
pred_keras = np.expm1(keras_nn.predict(X_test))
sub = pd.read_csv('../input/sample_submission.csv')
sub["target"] = pred_keras

print(sub.head())
sub.to_csv('submission.csv', index=False)
