# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from __future__ import absolute_import, division, print_function, unicode_literals



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pathlib



import matplotlib.pyplot as plt

import seaborn as sns



from scipy import stats

from scipy.stats import norm, skew #for some statistics



import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, Dense, Dropout, Flatten, LeakyReLU, MaxPooling2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.client import device_lib

import glob

# import imageio

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import pathlib

import os

from PIL import Image

import time

from IPython import display



print('TF version {ver}'.format(ver=tf.__version__))

print('Built with CUDA {cudaSupport} and GPU available {gpuAvailable}'.format(cudaSupport=tf.test.is_built_with_cuda(), gpuAvailable=tf.test.is_gpu_available()))

print(device_lib.list_local_devices())



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

print(os.listdir('../input/mercari-price-suggestion-challenge'))



# Any results you write to the current directory are saved as output.
DATA_BASE = '../input/mercari-price-suggestion-challenge'

EPOCHS = 10

BATCH_SIZE = 100000
df_train = pd.read_csv('../input/mercari-price-suggestion-challenge/train.tsv', sep='\t')

print(df_train.shape)

df_train.sample(10)
df_test = pd.read_csv('../input/mercari-price-suggestion-challenge/test.tsv', sep='\t')

print(df_test.shape)

df_train.sample(10)
df_train.groupby('brand_name').size().to_frame()
del df_train['train_id']

del df_train['name']

del df_train['item_description']
def prepare_data(df):

    df = df.replace(np.nan, 0).replace(np.inf, 1e+5).replace(-np.inf, -1e+5)

    for column in df.columns:

        if df[column].dtype.name == 'object':

            df[column] = pd.Categorical(df[column]).codes

            

        if column not in ['value']:

            col_stats = df[column].describe()

            df[column] = (df[column] - col_stats['mean']) / col_stats['std']

    return df
df_train_na = (df_train.isnull().sum() / len(df_train)) * 100

df_train_na = df_train_na.drop(df_train_na[df_train_na == 0].index).sort_values(ascending=False)[:30]

df_train_missing_data = pd.DataFrame({'Missing Ratio' :df_train_na})

df_train_missing_data.head(20)
df_train['brand_name'].fillna('**unknown**')

df_train['category_name'].fillna('**unknown**')

df_train['item_condition_id'] = df_train['item_condition_id'].astype(str)

df_train['shipping'] = df_train['shipping'].astype(str)

df_train["price"] = np.log1p(df_train["price"])
#Check the new distribution 

sns.distplot(df_train['price'] , fit=norm);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(df_train['price'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('price distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(df_train['price'], plot=plt)

plt.show()
df_train = prepare_data(df_train)

df_train.sample(10)
#Correlation map to see how features are correlated with price

corrmat = df_train.corr()

plt.subplots(figsize=(12,9))

sns.heatmap(corrmat, vmax=0.9, square=True)
fig, ax = plt.subplots()

ax.scatter(x = df_train['brand_name'], y = df_train['price'])

plt.ylabel('Price', fontsize=13)

plt.xlabel('BrandName', fontsize=13)

plt.show()
y_train = df_train.price.values

y_train = y_train.ravel().astype(np.float64)

df_train.drop('price', axis=1, inplace=True)
def build_model():

  model = tf.keras.Sequential([

    Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001), input_shape=[len(df_train.keys())]),

    Dropout(0.3),

    Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),

    Dropout(0.3),

    Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),

    Dropout(0.3),

    Dense(1)

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
print(df_train.dtypes)

df_train['item_condition_id'] = df_train['item_condition_id'].astype('float16')

df_train['category_name'] = df_train['category_name'].astype('float16')

df_train['brand_name'] = df_train['brand_name'].astype('float16')

df_train['shipping'] = df_train['shipping'].astype('float16')
# @tf.function

# def train_step(ds):

# #   losses = []

#   with tf.GradientTape() as gen_tape, tf.GradientTape() as tape:

#     real_output = model(ds[0], training=True)



#     loss = cross_entropy(ds[1], real_output)

#     losses.append(loss)



#   gradients = tape.gradient(loss, model.trainable_variables)

#   optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# #   return losses



# def train(ds, epochs):

#   for epoch in range(epochs):

#     start = time.time()

#     losses = []



# #     for i in range(int(DS_SIZE / BATCH_SIZE)):

# #       image_batch = next(iter(dataset))

#     batch_losses = train_step(ds)

#     losses.extend(batch_losses)



# #     checkpoint.step.assign_add(1)



#     print(f'Time for epoch {epoch} is {time.time()-start:.2f} sec, loss: {np.array(losses).mean():.6f}')



# dataset = tf.data.Dataset.from_tensor_slices((df_train.values, y_train))

# train(dataset, EPOCHS)
class MyProgbarLogger(tf.keras.callbacks.Callback):

  def on_train_begin(self, logs=None):

    self.seen = 0

    self.progbar = tf.keras.utils.Progbar(

        target=EPOCHS,

        unit_name='epoch')



  def on_epoch_end(self, epoch, logs=None):

    self.seen += 1

    self.progbar.update(self.seen)

    

class PrintDot(tf.keras.callbacks.Callback):

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

  plt.ylabel('Mean Abs Error [price]')

  plt.plot(hist['epoch'], hist['mae'],

           label='Train Error')

  plt.plot(hist['epoch'], hist['val_mae'],

           label = 'Val Error')

  plt.ylim([0,1])

  plt.legend()



  plt.figure()

  plt.xlabel('Epoch')

  plt.ylabel('Mean Square Error [$price^2$]')

  plt.plot(hist['epoch'], hist['mse'],

           label='Train Error')

  plt.plot(hist['epoch'], hist['val_mse'],

           label = 'Val Error')

  plt.ylim([0,1])

  plt.legend()

  plt.show()





plot_history(history)