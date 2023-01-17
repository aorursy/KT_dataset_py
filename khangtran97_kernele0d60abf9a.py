import numpy as np # linear algebra

# np.random.seed(8)

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))



from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.preprocessing import MinMaxScaler, StandardScaler, scale

from sklearn.metrics import roc_auc_score



from keras.layers import Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D, BatchNormalization, Input, Conv2D

from keras.callbacks import ModelCheckpoint

from keras import metrics

from keras.optimizers import Adam 

from keras import backend as K

import keras

from keras.models import Model

from keras import Sequential

from keras import regularizers

import tensorflow as tf

from keras.losses import binary_crossentropy

import gc

import scipy.special

from tqdm import *

from scipy.stats import norm, rankdata



from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau



train = pd.read_csv('../input/dataset/Train_full.csv')

test = pd.read_csv('../input/dataset-v2/Test_small_features.csv')
train.shape
train.head()
test.shape
test.head()
arr = []

for i in range(test.shape[0]):

    if i == 0:

        continue

    else:

        if test.at[i, 'body'] > 0:

            arr.append(1)

        else:

            arr.append(0)

arr.append(0)            

y_true = pd.DataFrame(arr)
all_data = pd.concat((train.loc[:,'Open':'lag_return_96'],

                      test.loc[:,'Open':'lag_return_96']))

all_data.head()
all_data.head()
all_data.shape
cat_feat = ['hour', 'min', 'dayofweek']
hour = pd.get_dummies(all_data['hour'])

min_data = pd.get_dummies(all_data['min'])

day_data = pd.get_dummies(all_data['dayofweek'])
all_data = all_data.drop(cat_feat,axis = 1)
for i in hour.columns:

    hour.rename(columns={i:'hour_%s'%i}, inplace=True)
for i in min_data.columns:

    min_data.rename(columns={i:'min_%s'%i}, inplace=True)
for i in day_data.columns:

    day_data.rename(columns={i:'day_%s'%i}, inplace=True)
all_data.shape
all_data = pd.concat([all_data, hour, min_data, day_data], axis=1)
all_data.shape
features = all_data.columns
# Feature Scaling

sc = MinMaxScaler()

# x_train = sc.fit_transform(x_train)

# x_test = sc.transform(x_test)

# test_features = sc.transform(test_features)

scaled_df = sc.fit_transform(all_data)

all_data = pd.DataFrame(scaled_df, columns = features)
all_data.head()
train_features = all_data[:train.shape[0]]

train_targets = train['up_down']

test_features = all_data[train.shape[0]:]
x_train, x_test, y_train, y_test = train_test_split(train_features, train_targets, test_size = 0.2, random_state = 50, shuffle = False)
def auc(y_true, y_pred):

    auc = tf.metrics.auc(y_true, y_pred)[1]

    K.get_session().run(tf.local_variables_initializer())

    return auc
input_dim = x_train.shape[1]

input_dim
import tensorflow as tf

import pandas as pd

import os

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from keras import Sequential

from keras import layers

from keras import backend as K

from keras.layers.core import Dense

from keras import regularizers

from keras.layers import Dropout

from keras.constraints import max_norm
model = Sequential()

# Input layer

model.add(Dense(units = 512, activation = "relu", input_dim = input_dim, kernel_initializer = "normal", kernel_regularizer=regularizers.l2(0.005), 

                kernel_constraint = max_norm(5.)))

# Add dropout regularization

model.add(Dropout(rate=0.4))



# Second hidden layer

model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.005), kernel_constraint=max_norm(5)))

# Add dropout regularization

model.add(Dropout(rate=0.35))



# Third hidden layer

model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.005), kernel_constraint=max_norm(5)))

# Add dropout regularization

model.add(Dropout(rate=0.3))



# Output layer

model.add(layers.Dense(units = 1, activation='sigmoid'))



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', auc])

model.summary()
checkpoint = ModelCheckpoint('feed_forward_model.h5', monitor='val_loss', verbose=1, 

                             save_best_only=True, mode='min', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, 

                                   verbose=1, mode='min', epsilon=0.0001)

early = EarlyStopping(monitor="val_loss", 

                      mode="min", 

                      patience=10)
callbacks_list = [early, checkpoint, reduceLROnPlat]
model.fit(x_train, y_train, batch_size = 166384, epochs = 125, validation_data = (x_test, y_test), callbacks = callbacks_list)
model.load_weights('feed_forward_model.h5')

prediction = model.predict(test_features, batch_size=512, verbose=1)
arr_test = pd.DataFrame(prediction)
arr_test.head()
pred = []

for i in range(test_features.shape[0]):

    if arr_test.at[i, 0] > 0.475:

        pred.append(1)

    else:

        pred.append(0)

        

y_pred = pd.DataFrame(pred)
test_features.shape
from sklearn.metrics import accuracy_score

accuracy_score(y_pred, y_true)
model.load_weights('feed_forward_model.h5')

prediction = model.predict(test_features, batch_size=512, verbose=1)
arr_test = pd.DataFrame(prediction)
pred = []

for i in range(test_features.shape[0]):

    if arr_test.at[i, 0] > 0.51:

        pred.append(1)

    else:

        pred.append(0)

        

y_pred = pd.DataFrame(pred)

from sklearn.metrics import accuracy_score

accuracy_score(y_pred, y_true)