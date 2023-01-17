import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.keras import callbacks
from IPython.display import display

import matplotlib.pyplot as plt

import seaborn as sns



plt.style.use('seaborn-whitegrid')

# Set Matplotlib defaults

plt.rc('figure', autolayout=True)

plt.rc('axes', labelweight='bold', labelsize='large',

       titleweight='bold', titlesize=18, titlepad=10)
concrete = pd.read_csv('../input/dl-course-data/concrete.csv')
display(concrete.head())

display(concrete.info())

display(concrete.describe())
sns.pairplot(concrete);
df = concrete.copy()



df_train = df.sample(frac=0.7, random_state=0)

df_valid = df.drop(df_train.index)



mean = df_train.mean(axis=0)

std = df_train.std(axis=0)

df_train = (df_train - mean) / std

df_valid = (df_valid - mean) / std



df_train.describe()
X_train = df_train.drop('CompressiveStrength', axis=1)

X_valid = df_valid.drop('CompressiveStrength', axis=1)

y_train = df_train['CompressiveStrength']

y_valid = df_valid['CompressiveStrength']



ds_train_ = tf.data.Dataset.from_tensor_slices((X_train, y_train))

ds_valid_ = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))



BATCH_SIZE = 32

NUM_FEATURES = len(X_train.keys())

AUTO = tf.data.experimental.AUTOTUNE

ds_train = (ds_train_

            .batch(BATCH_SIZE)

            .cache()

            .shuffle(10000)

            .prefetch(AUTO))



ds_valid = (ds_valid_

            .batch(BATCH_SIZE)

            .cache()

            .prefetch(AUTO))
model = keras.Sequential([

    layers.InputLayer(input_shape=(NUM_FEATURES, )),

    layers.Dense(1),

])

model.compile(

    optimizer='sgd',

    loss='mae',

    metrics=['mae', 'mse'],

)
early_stopping = keras.callbacks.EarlyStopping(patience=10, min_delta=0.0001)

EPOCHS = 1000

history = model.fit(

    ds_train,

    validation_data=ds_valid,

    epochs=EPOCHS,

    callbacks=[early_stopping],

    verbose=0,

)

# -
history_df = pd.DataFrame(history.history)

history_df.loc[2:, ['loss', 'val_loss']].plot();
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()))
model = keras.Sequential([

    layers.InputLayer(input_shape=(NUM_FEATURES, )),

    layers.Dense(8, activation='relu'),

    layers.Dense(8, activation='relu'),

    layers.Dense(1),

])

model.compile(

    optimizer='adam',

    loss='mae',

    metrics=['mae'],

)
early_stopping = keras.callbacks.EarlyStopping(patience=20, min_delta=1e-4)

EPOCHS = 1000

history = model.fit(

    ds_train,

    validation_data=ds_valid,

    epochs=EPOCHS,

    callbacks=[early_stopping],

    verbose=0,

)

# -
history_df = pd.DataFrame(history.history)

history_df.loc[2:, ['loss', 'val_loss']].plot();
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()))
model = keras.Sequential([

    layers.InputLayer(input_shape=(NUM_FEATURES, )),

    layers.Dense(512, activation='relu'),

    layers.Dense(512, activation='relu'),    

    layers.Dense(512, activation='relu'),

    layers.Dense(1),

])

model.compile(

    optimizer='adam',

    loss='mae',

    metrics=['mae'],

)
early_stopping = keras.callbacks.EarlyStopping(patience=20, min_delta=1e-4)

EPOCHS = 1000

history = model.fit(

    ds_train,

    validation_data=ds_valid,

    epochs=EPOCHS,

    callbacks=[early_stopping],

    verbose=0,

)

# -
history_df = pd.DataFrame(history.history)

history_df.loc[0:, ['loss', 'val_loss']].plot();
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()))
model = keras.Sequential([

    layers.InputLayer(input_shape=(NUM_FEATURES, )),

    layers.Dense(512, activation='relu'),

    layers.Dropout(0.2),

    layers.Dense(512, activation='relu'),

    layers.Dropout(0.2),

    layers.Dense(512, activation='relu'),

    layers.Dropout(0.2),

    layers.Dense(1),

])

model.compile(

    optimizer='adam',

    loss='mae',

    metrics=['mae'],

)
early_stopping = keras.callbacks.EarlyStopping(patience=20, min_delta=1e-4)

EPOCHS = 1000

history = model.fit(

    ds_train,

    validation_data=ds_valid,

    epochs=EPOCHS,

    callbacks=[early_stopping],

    verbose=0,

)

# -
history_df = pd.DataFrame(history.history)

history_df.loc[2:, ['loss', 'val_loss']].plot();
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()))
model = keras.Sequential([

    layers.InputLayer(input_shape=(NUM_FEATURES, )),

    layers.Dense(512),

    layers.Activation('relu'),

    layers.BatchNormalization(),

    layers.Dense(512),

    layers.Activation('relu'),

    layers.BatchNormalization(),

    layers.Dense(512),

    layers.Activation('relu'),

    layers.BatchNormalization(),

    layers.Dense(1),

])

model.compile(

    optimizer='adam',

    loss='mae',

    metrics=['mae'],

)
early_stopping = keras.callbacks.EarlyStopping(patience=20, min_delta=1e-4)

EPOCHS = 1000

history = model.fit(

    ds_train,

    validation_data=ds_valid,

    epochs=EPOCHS,

    callbacks=[early_stopping],

    verbose=0,

)

# -
history_df = pd.DataFrame(history.history)

history_df.loc[0:, ['loss', 'val_loss']].plot();
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()))