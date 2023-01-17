# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pathlib



import matplotlib.pyplot as plt

import seaborn as sns



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers



from scipy import stats

from scipy.stats import norm, skew #for some statistics



# print(f'TF version {tf.__version__}')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

print(os.listdir("../input/cat-in-the-dat/"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/cat-in-the-dat/train.csv')

df_test = pd.read_csv('../input/cat-in-the-dat/test.csv')



print(f'Train shape {df_train.shape}')

print(f'Test shape {df_test.shape}')
df_sample = pd.read_csv('../input/cat-in-the-dat/sample_submission.csv')

df_sample.sample(10)
print(df_train.sample(10))
df_train.sample(10)
df_test.sample(10)
y_train = df_train.target.values

y_train = y_train.ravel().astype(np.float64)

df_train.drop(['target'], axis=1, inplace=True)

df_train.drop(['id'], axis=1, inplace=True)



df_id = df_test.id.values

df_test.drop(['id'], axis=1, inplace=True)

df_all = pd.concat((df_train, df_test), sort=False).reset_index(drop=True)
all_na = (df_all.isnull().sum() / len(df_all)) * 100

all_na = all_na.drop(all_na[all_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :all_na})

missing_data.head(20)
def prepare_data(df):

    df = df.replace(np.nan, 0).replace(np.inf, 1e+5).replace(-np.inf, -1e+5)

    for column in df.columns:

        if df[column].dtype.name == 'object' and column != 'timestamp':

            df[column] = pd.Categorical(df[column]).codes

        if column not in ['site_id', 'building_id', 'timestamp', 'meter', 'meter_reading']:

            col_stats = df[column].describe()

            df[column] = (df[column] - col_stats['mean']) / col_stats['std']

    return df



df_all = prepare_data(df_all)

df_all.sample(10)
df_train = df_all[:df_train.shape[0]]

df_test = df_all[df_train.shape[0]:]
def build_model():

  model = keras.Sequential([

    layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001), input_shape=[len(df_train.keys())]),

    layers.Dropout(0.3),

    layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),

    layers.Dropout(0.3),

    layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),

    layers.Dropout(0.3),

    layers.Dense(1)

  ])



  optimizer = tf.keras.optimizers.RMSprop(1e-4)

#   optimizer = tf.keras.optimizers.Adam(1e-4)



#   model.compile(loss='mse',

#                 optimizer=optimizer,

#                 metrics=['mae', 'mse'])



  model.compile(loss='mse',

                optimizer=optimizer,

                metrics=['mae', 'mse'])

  return model



model = build_model()



model.summary()
EPOCHS = 30



class MyProgbarLogger(keras.callbacks.Callback):

  def on_train_begin(self, logs=None):

    self.seen = 0

    self.progbar = keras.utils.Progbar(

        target=EPOCHS,

        unit_name='epoch')



  def on_epoch_end(self, epoch, logs=None):

    self.seen += 1

    self.progbar.update(self.seen)

    

class PrintDot(keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs=None):

    print('.', end='')

    

# progbar = keras.callbacks.ProgbarLogger(params={'verbose': False})

# early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)



history = model.fit(

  df_train, y_train,

  epochs=EPOCHS, validation_split = 0.2, verbose=0,

#   callbacks=[early_stop, MyProgbarLogger()])

    callbacks=[MyProgbarLogger()])
def plot_history(history):

  hist = pd.DataFrame(history.history)

  hist['epoch'] = history.epoch



  plt.figure()

  plt.xlabel('Epoch')

  plt.ylabel('Mean Abs Error [meter_reading]')

  plt.plot(hist['epoch'], hist['mae'],

           label='Train Error')

  plt.plot(hist['epoch'], hist['val_mae'],

           label = 'Val Error')

  plt.ylim([0,10])

  plt.legend()



  plt.figure()

  plt.xlabel('Epoch')

  plt.ylabel('Mean Square Error [$meter_reading^2$]')

  plt.plot(hist['epoch'], hist['mse'],

           label='Train Error')

  plt.plot(hist['epoch'], hist['val_mse'],

           label = 'Val Error')

  plt.ylim([0,10])

  plt.legend()

  plt.show()





plot_history(history)
loss, mae, mse = model.evaluate(df_train, y_train, verbose=2)



print("Testing set Mean Abs Error: {:5.2f} meter_reading".format(mae))
test_predictions = model.predict(df_train).flatten()



plt.scatter(y_train, test_predictions)

plt.xlabel('True Values [meter_reading]')

plt.ylabel('Predictions [meter_reading]')

plt.axis('equal')

plt.axis('square')

plt.xlim([0,plt.xlim()[1]])

plt.ylim([0,plt.ylim()[1]])

_ = plt.plot([-100, 100], [-100, 100])
y_pred = model.predict(df_test).flatten().ravel()

sub = pd.DataFrame()

sub["id"] = df_id

sub["target"] = y_pred

print(sub.head(20))

sub.to_csv('submission.csv', index=False)