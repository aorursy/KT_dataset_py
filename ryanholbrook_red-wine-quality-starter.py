import pandas as pd



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers



from IPython.display import display

import seaborn as sns

import matplotlib.pyplot as plt



plt.style.use('seaborn-whitegrid')

plt.rc('figure', autolayout=True)

plt.rc('axes', labelweight='bold', labelsize='large',

       titleweight='bold', titlesize=18, titlepad=10)
red_wine = pd.read_csv('../input/dl-course-data/red-wine.csv')

display(red_wine.head())

display(red_wine.info())

display(red_wine.describe())
sns.pairplot(red_wine);
df_train = red_wine.sample(frac=0.7, random_state=0)

df_valid = red_wine.drop(df_train.index)



mean = df_train.mean(axis=0)

std = df_train.std(axis=0)

df_train = (df_train - mean) / std

df_valid = (df_valid - mean) / std



df_train.describe()
X_train = df_train.drop('quality', axis=1)

X_valid = df_valid.drop('quality', axis=1)

y_train = df_train['quality']

y_valid = df_valid['quality']



ds_train_ = tf.data.Dataset.from_tensor_slices((X_train, y_train))

ds_valid_ = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))



BATCH_SIZE = 256

NUM_FEATURES = len(X_train.keys())

AUTO = tf.data.experimental.AUTOTUNE

ds_train = (ds_train_

            .cache()

            .shuffle(1000)

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

    layers.Dense(16, activation='relu'),

    layers.Dense(16, activation='relu'),

    layers.Dense(16, activation='relu'),

    layers.Dense(1),

])

model.compile(

    optimizer='adam',

    loss='mae',

    metrics=['mae'],

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
history_df = pd.DataFrame(history.history)

history_df.loc[2:, ['loss', 'val_loss']].plot();
print("Minimum Validation Loss: {:0.4f}".format(history_df['loss'].min()))
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
EPOCHS = 200

history = model.fit(

    ds_train,

    validation_data=ds_valid,

    epochs=EPOCHS,

    verbose=0,

)
history_df = pd.DataFrame(history.history)

history_df.loc[0:, ['loss', 'val_loss']].plot();
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()))
model = keras.Sequential([

    layers.InputLayer(input_shape=(NUM_FEATURES, )),

    layers.Dense(512, activation='relu'),

    layers.Dropout(0.5),

    layers.Dense(512, activation='relu'),

    layers.Dropout(0.5),

    layers.Dense(512, activation='relu'),

    layers.Dropout(0.5),

    layers.Dense(1),

])

model.compile(

    optimizer='adam',

    loss='mae',

    metrics=['mae'],

)
EPOCHS = 200

history = model.fit(

    ds_train,

    validation_data=ds_valid,

    epochs=EPOCHS,

    verbose=0,

)
history_df = pd.DataFrame(history.history)

history_df.loc[2:, ['loss', 'val_loss']].plot();
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()))
reg = keras.regularizers.l2(0.0003)



model = keras.Sequential([

    layers.InputLayer(input_shape=(NUM_FEATURES, )),

    layers.Dense(2048, kernel_regularizer=reg),

    layers.BatchNormalization(),

    layers.Activation('relu'),

    layers.Dropout(0.3),

    layers.Dense(2048, kernel_regularizer=reg),

    layers.BatchNormalization(),

    layers.Activation('relu'),

    layers.Dropout(0.3),

    layers.Dense(1024, kernel_regularizer=reg),

    layers.BatchNormalization(),

    layers.Activation('relu'),

    layers.Dropout(0.3),

    layers.Dense(1, kernel_regularizer=reg),

])

model.compile(

    optimizer=keras.optimizers.Adam(0.001),

    loss='mae',

    metrics=['mae', 'mse'],

)



EPOCHS = 500

early_stopping = keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)

lr_schedule = keras.callbacks.ReduceLROnPlateau(patience=10)

history = model.fit(

    ds_train,

    validation_data=ds_valid,

    epochs=EPOCHS,

    callbacks=[early_stopping, lr_schedule],

    verbose=0,

)
history_df = pd.DataFrame(history.history)

history_df.loc[2:, ['loss', 'val_loss']].plot()

history_df.loc[2:, ['mae', 'val_mae']].plot();
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()))