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
import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split



import tensorflow as tf
# random seed fixed

np.random.seed(5)

tf.random.set_seed(5)
train = pd.read_csv('/kaggle/input/dacon-stage1/train.csv')

train.head(3)
test = pd.read_csv('/kaggle/input/dacon-stage1/test.csv')

test = test.drop('id', axis=1)

test.head(3)
plt.figure(figsize=(18, 5))

sns.countplot(x=train['layer_1'])
plt.figure(figsize=(18, 5))

sns.countplot(x=train['layer_2'])
plt.figure(figsize=(18, 5))

sns.countplot(x=train['layer_3'])
plt.figure(figsize=(18, 5))

sns.countplot(x=train['layer_4'])
plt.figure(figsize=(20, 15))

sns.heatmap(train.corr())
plt.figure(figsize=(18, 5))

sns.boxplot(train['layer_1'], train['1'])
plt.figure(figsize=(18, 5))

sns.boxplot(train['layer_1'], train['100'])
plt.figure(figsize=(18, 5))

sns.boxplot(train['layer_1'], train['200'])
plt.figure(figsize=(18, 5))

sns.boxplot(train['layer_1'], train['225'])
train.describe().T
train.groupby('layer_1').mean().T
train.groupby('layer_1').max().T
train.groupby('layer_1').min().T
# using keras model

# train data -> train / validation split. case 1.



X = train.drop(['layer_1', 'layer_2', 'layer_3', 'layer_4'], axis=1)

Y = train[['layer_1', 'layer_2', 'layer_3', 'layer_4']]



X_train_1, X_vali_1, Y_train_1, Y_vali_1 = train_test_split(X, Y, random_state=1, test_size=0.05)

X_train_1.shape, Y_train_1.shape, X_vali_1.shape, Y_vali_1.shape
# train data -> train / validation split. case 2.

X_train_2, X_vali_2, Y_train_2, Y_vali_2 = train_test_split(X, Y, random_state=2, test_size=0.05)

X_train_2.shape, Y_train_2.shape, X_vali_2.shape, Y_vali_2.shape
# train data -> train / validation split. case 3.

X_train_3, X_vali_3, Y_train_3, Y_vali_3 = train_test_split(X, Y, random_state=3, test_size=0.05)

X_train_3.shape, Y_train_3.shape, X_vali_3.shape, Y_vali_3.shape
import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, BatchNormalization, LeakyReLU

from keras.optimizers import adam

from keras.callbacks import ReduceLROnPlateau

from keras.callbacks import ModelCheckpoint



import warnings

warnings.filterwarnings('ignore')
# Generate Keras Model. case 1.

model_1 = Sequential()

model_1.add(Dense(units=2048, input_dim=226))

model_1.add(BatchNormalization())

model_1.add(LeakyReLU(alpha=0.1))

model_1.add(Dense(units=2048))

model_1.add(BatchNormalization())

model_1.add(LeakyReLU(alpha=0.1))

model_1.add(Dense(units=2048))

model_1.add(BatchNormalization())

model_1.add(LeakyReLU(alpha=0.1))

model_1.add(Dense(units=2048))

model_1.add(BatchNormalization())

model_1.add(LeakyReLU(alpha=0.1))

model_1.add(Dense(units=2048))

model_1.add(BatchNormalization())

model_1.add(LeakyReLU(alpha=0.1))

model_1.add(Dense(units=4))

model_1.summary()
# Generate Keras Model. case 2.

model_2 = Sequential()

model_2.add(Dense(units=2048, input_dim=226))

model_2.add(BatchNormalization())

model_2.add(LeakyReLU(alpha=0.1))

model_2.add(Dense(units=2048))

model_2.add(BatchNormalization())

model_2.add(LeakyReLU(alpha=0.1))

model_2.add(Dense(units=2048))

model_2.add(BatchNormalization())

model_2.add(LeakyReLU(alpha=0.1))

model_2.add(Dense(units=2048))

model_2.add(BatchNormalization())

model_2.add(LeakyReLU(alpha=0.1))

model_2.add(Dense(units=2048))

model_2.add(BatchNormalization())

model_2.add(LeakyReLU(alpha=0.1))

model_2.add(Dense(units=4))

model_2.summary()
# Generate Keras Model. case 3.

model_3 = Sequential()

model_3.add(Dense(units=2048, input_dim=226))

model_3.add(BatchNormalization())

model_3.add(LeakyReLU(alpha=0.1))

model_3.add(Dense(units=2048))

model_3.add(BatchNormalization())

model_3.add(LeakyReLU(alpha=0.1))

model_3.add(Dense(units=2048))

model_3.add(BatchNormalization())

model_3.add(LeakyReLU(alpha=0.1))

model_3.add(Dense(units=2048))

model_3.add(BatchNormalization())

model_3.add(LeakyReLU(alpha=0.1))

model_3.add(Dense(units=2048))

model_3.add(BatchNormalization())

model_3.add(LeakyReLU(alpha=0.1))

model_3.add(Dense(units=4))

model_3.summary()
# opti = keras.optimizers.Adam(lr=0.002)

# model.compile(loss='mae', optimizer=opti, metrics=['mae'])
# model compile

model_1.compile(loss='mae', optimizer='adam', metrics=['mae'])

model_2.compile(loss='mae', optimizer='adam', metrics=['mae'])

model_3.compile(loss='mae', optimizer='adam', metrics=['mae'])
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,

#                               patience=3, min_lr=0.000125)
check_best_1 = keras.callbacks.ModelCheckpoint(filepath='best_model_1_ep250', monitor='val_loss',

verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)



check_best_2 = keras.callbacks.ModelCheckpoint(filepath='best_model_2_ep250', monitor='val_loss',

verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)



check_best_3 = keras.callbacks.ModelCheckpoint(filepath='best_model_3_ep250', monitor='val_loss',

verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
from keras.callbacks import LearningRateScheduler

import math

# learning rate schedule

def step_decay(epoch):

	initial_lrate = 0.001

	drop = 0.2

	epochs_drop = 60

	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))

	return lrate



# learning schedule callback

lrate = LearningRateScheduler(step_decay)
callback_list_1 = [lrate, check_best_1]

callback_list_2 = [lrate, check_best_2]

callback_list_3 = [lrate, check_best_3]
# model_1 training

history_1 = model_1.fit(X_train_1, Y_train_1, epochs=5, batch_size=256,

                        validation_data=(X_vali_1, Y_vali_1), callbacks=callback_list_1)

# Originally used epochs = 250
# model_2 training

history_2 = model_2.fit(X_train_2, Y_train_2, epochs=5, batch_size=256,

                        validation_data=(X_vali_2, Y_vali_2), callbacks=callback_list_2)

# Originally used epochs = 250
# model_3 training

history_3 = model_3.fit(X_train_3, Y_train_3, epochs=5, batch_size=256,

                        validation_data=(X_vali_3, Y_vali_3), callbacks=callback_list_3)

# Originally used epochs = 250
# Plot training & validation loss values

plt.figure(figsize=(10,7))

plt.plot(history_1.history['loss'])

plt.plot(history_1.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
# best model loading

from tensorflow.keras.models import load_model

best_model_1 = load_model('best_model_1_ep250')

best_model_2 = load_model('best_model_2_ep250')

best_model_3 = load_model('best_model_3_ep250')



# validation mae

pred_vali_1 = best_model_1.predict(X_vali_1)

pred_vali_2 = best_model_2.predict(X_vali_2)

pred_vali_3 = best_model_3.predict(X_vali_3)

print(mean_absolute_error(Y_vali_1, pred_vali_1))

print(mean_absolute_error(Y_vali_2, pred_vali_2))

print(mean_absolute_error(Y_vali_3, pred_vali_3))
# predict test set

pred_test_1 = best_model_1.predict(test)

pred_test_2 = best_model_2.predict(test)

pred_test_3 = best_model_3.predict(test)
# load submission file

sample_sub = pd.read_csv('/kaggle/input/dacon-stage1/sample_submission.csv', index_col=0)

sample_sub.head(3)
# blend the predictions.

sub = sample_sub + (0.34*pred_test_1 + 0.33*pred_test_2 + 0.33*pred_test_3)

sub.head()
# post-processing

print((sub['layer_1'] < 10).sum())

print((sub['layer_2'] < 10).sum())

print((sub['layer_3'] < 10).sum())

print((sub['layer_4'] < 10).sum())

print((sub['layer_1'] > 300).sum())

print((sub['layer_2'] > 300).sum())

print((sub['layer_3'] > 300).sum())

print((sub['layer_4'] > 300).sum())
sub.loc[sub['layer_1'] < 10, 'layer_1'] = 10

sub.loc[sub['layer_2'] < 10, 'layer_2'] = 10

sub.loc[sub['layer_3'] < 10, 'layer_3'] = 10

sub.loc[sub['layer_4'] < 10, 'layer_4'] = 10



sub.loc[sub['layer_1'] > 300, 'layer_1'] = 300

sub.loc[sub['layer_2'] > 300, 'layer_2'] = 300

sub.loc[sub['layer_3'] > 300, 'layer_3'] = 300

sub.loc[sub['layer_4'] > 300, 'layer_4'] = 300



print((sub['layer_1'] < 10).sum())

print((sub['layer_2'] < 10).sum())

print((sub['layer_3'] < 10).sum())

print((sub['layer_4'] < 10).sum())

print((sub['layer_1'] > 300).sum())

print((sub['layer_2'] > 300).sum())

print((sub['layer_3'] > 300).sum())

print((sub['layer_4'] > 300).sum())
# save the csv file

sub.to_csv('dacon_stage_1_submission_pred_test_post_processing.csv')