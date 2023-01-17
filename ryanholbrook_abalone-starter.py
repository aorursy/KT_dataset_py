import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.keras import callbacks

from IPython.display import display

import matplotlib.pyplot as plt

import seaborn as sns
abalone = pd.read_csv('../input/dl-course-data/abalone.csv')
display(abalone.head())

display(abalone.info())

display(abalone.describe())
sns.pairplot(abalone);
df = abalone.copy()

df = pd.get_dummies(df, drop_first=True)



df_train = df.sample(frac=0.7, random_state=0)

df_valid = df.drop(df_train.index)



max_ = df_train.max(axis=0)

min_ = df_train.min(axis=0)



df_train = (df_train - min_) / (max_ - min_)

df_valid = (df_valid - min_) / (max_ - min_)



df_train.describe()
X_train = df_train.drop('Rings', axis=1)

X_valid = df_valid.drop('Rings', axis=1)

y_train = df_train['Rings']

y_valid = df_valid['Rings']



ds_train_ = tf.data.Dataset.from_tensor_slices((X_train, y_train))

ds_valid_ = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))



BATCH_SIZE = 512

NUM_FEATURES = len(X_train.keys())

AUTO = tf.data.experimental.AUTOTUNE

ds_train = (ds_train_

            .cache()

            .shuffle(10000)

            .batch(BATCH_SIZE)

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

    metrics=['mae'],

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

    metrics=['mse'],

)
early_stopping = keras.callbacks.EarlyStopping(patience=20, min_delta=1e-5)

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

history_df.loc[10:, ['loss', 'val_loss']].plot();
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

    metrics=['mse'],

)
early_stopping = keras.callbacks.EarlyStopping(patience=20, min_delta=1e-5)

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

    layers.Dropout(0.1),

    layers.Dense(512, activation='relu'),

    layers.Dropout(0.1),

    layers.Dense(512, activation='relu'),

    layers.Dropout(0.1),

    layers.Dense(1),

])

initial_learning_rate = 0.001

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(

    initial_learning_rate,

    decay_steps=10,

    decay_rate=0.9,

)

model.compile(

    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),

    loss='mae',

    metrics=['mse'],

)
early_stopping = keras.callbacks.EarlyStopping(patience=20, min_delta=1e-4, restore_best_weights=True)

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

history_df.loc[0:, ['loss', 'val_loss']].plot()

history_df.loc[0:, ['mse', 'val_mse']].plot();
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()))