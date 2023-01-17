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
# util

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

# deep learning

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers, models

from keras import regularizers

from keras.wrappers.scikit_learn import KerasClassifier

from keras.callbacks import EarlyStopping

# sklearn e preprocess

from sklearn import preprocessing

from sklearn.preprocessing import MinMaxScaler,  RobustScaler

from sklearn.model_selection import GridSearchCV

# seed

from numpy.random import seed

##

## Data Load

##

train_data = pd.read_csv('/kaggle/input/desafio-worcap-2020/treino.csv', )

test_data = pd.read_csv('/kaggle/input/desafio-worcap-2020/teste.csv')
train_data.describe()
##

## Data Split

##

train_x = train_data.drop('id', axis=1)

train_x = train_x.loc[:, train_x.columns != 'label'] 

train_y = train_data.loc[:,'label'].values 

#

test_x = test_data.drop('id', axis=1)

test_x = test_x.loc[:, test_x.columns != 'label'] 

print(train_x.shape, train_y.shape)

print(test_x.shape, )

num_classes = len(set(train_data['label'].values))

print(num_classes,' classes')

n_features = train_x.shape[1]

print(n_features,' features')
##

## Data Categorical-->OneHot Encoder

##

encoder = preprocessing.OneHotEncoder(sparse=False)



encoder.fit(train_y.reshape(-1, 1))

train_y_onehot = encoder.transform(train_y.reshape(-1, 1))

print(train_y_onehot[:5])

print(train_y[:5])

print(encoder.inverse_transform(train_y_onehot[:5]))
##

## Data Scale

##

transformer = RobustScaler([0,1]).fit(train_x)

#transformer = MinMaxScaler([0,1]).fit(train_x)

#

train_x_scaled = transformer.transform(train_x)

test_x_scaled = transformer.transform(test_x)
##

## Data Reshape to Conv1D Layers 

##

print('Original Train Shape:', train_x_scaled.shape)

print('Original Test Shape:', test_x_scaled.shape)

# Shape to Conv1D

train_x_conv1d = train_x_scaled.reshape(train_x_scaled.shape[0],train_x_scaled.shape[1], 1)

test_x_conv1d = test_x_scaled.reshape(test_x_scaled.shape[0],test_x_scaled.shape[1], 1)

print('#######')

print('Train shape to Conv1D', train_x_conv1d.shape)

print('Test shape to Conv1D', test_x_conv1d.shape)
##

## Best Model (189 epochs)

## 

tf.random.set_seed(7)

seed(7)

#

conv_model = models.Sequential([

    layers.Conv1D(32, 5, activation='relu', input_shape=(n_features,1)),

    layers.MaxPooling1D(pool_size=2, strides=None, padding="valid",),

    layers.Conv1D(16, 3, activation='relu', input_shape=(n_features,1)),

    layers.MaxPooling1D(pool_size=2, strides=None, padding="valid",),

    layers.Flatten(),

    layers.Dense(16, activation='relu', ),

    layers.Dense(num_classes, activation='softmax',),

]) 

lr = 0.001

conv_model.compile(optimizer=keras.optimizers.Adam(lr),

              loss=keras.losses.categorical_crossentropy,

              metrics=['accuracy'])

conv_model.summary()
##

## Best Model Train 

## 

# Callback 

es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=100,

                          restore_best_weights=False)

# Train with Validation  (20% validation)

history = conv_model.fit(train_x_conv1d, train_y_onehot, epochs=400, verbose=0, 

                         batch_size=len(train_x), callbacks=[es], validation_split=0.2)

# Results

acc = '{:.4}'.format(100*history.history['accuracy'][-1])

val_acc = '{:.4}'.format(100*history.history['val_accuracy'][-1])

print('Train Accuracy:', acc)

print('Validation Accuracy:', val_acc)

# Plot Acc Curve

plt.figure(figsize=(12,4))

plt.plot(history.history['accuracy'], label='Train Acc '+acc)

plt.plot(history.history['val_accuracy'], label='Validation Acc '+val_acc)

plt.legend()

plt.show()

# Plot loss

plt.figure(figsize=(12,4))

plt.plot(history.history['loss'], 'r', label='Loss')

plt.legend()

plt.show()
##

## Best Model 

## 

tf.random.set_seed(7)

seed(7)

#

final_conv_model = models.Sequential([

    layers.Conv1D(32, 5, activation='relu', input_shape=(n_features,1)),

    layers.MaxPooling1D(pool_size=2, strides=None, padding="valid",),

    layers.Conv1D(16, 3, activation='relu', input_shape=(n_features,1)),

    layers.MaxPooling1D(pool_size=2, strides=None, padding="valid",),

    layers.Flatten(),

    layers.Dense(16, activation='relu', ),

    layers.Dense(num_classes, activation='softmax',),

]) 

lr = 0.001

final_conv_model.compile(optimizer=keras.optimizers.Adam(lr),

              loss=keras.losses.categorical_crossentropy,

              metrics=['accuracy'])
##

## Best Model Train 

## 

best_epoch_num = 256 # may vary

# Train without Validation and Callbacks

history = final_conv_model.fit(train_x_conv1d, train_y_onehot, epochs=best_epoch_num, 

                               verbose=0, batch_size=len(train_x),)

# Results

acc = '{:.4}'.format(100*history.history['accuracy'][-1])

print('Train Accuracy:', acc)

# Plot Acc Curve

plt.figure(figsize=(12,4))

plt.plot(history.history['accuracy'], label='Train Acc '+acc)

plt.legend()

plt.show()

# Plot loss

plt.figure(figsize=(12,4))

plt.plot(history.history['loss'], 'r', label='Loss')

plt.legend()

plt.show()
##

## Best Model Predict 

## 

y_pred = final_conv_model.predict(test_x_conv1d)

y_pred = encoder.inverse_transform(y_pred).reshape(-1)

#y_pred = [x.strip(' ') for x in y_pred]

print(y_pred[:10])

output = pd.DataFrame({'id': test_data.id,'label': y_pred})

output.to_csv('result.csv', index=False)

pd.read_csv('result.csv', squeeze=True)
##

## Make Model ( 2 conv1d + 2 maxpooling + 1 dense + output dense layer)

## 

def create_model_from_params(params={'conv1_filters':32,

                                             'conv2_filters':16,

                                             'conv1_kernel':5,

                                             'conv2_kernel':3,

                                             'dense_neur':16,

                                             'maxp1':2,

                                             'maxp2':2}):

    tf.random.set_seed(7)

    seed(7)

    # rede conv

    m = models.Sequential([

        layers.Conv1D(params['conv1_filters'], params['conv1_kernel'], activation='relu', input_shape=(n_features,1)),

        layers.MaxPooling1D(pool_size=params['maxp1'], strides=None, padding="valid",),

        layers.Conv1D(params['conv2_filters'], params['conv2_kernel'], activation='relu', input_shape=(n_features,1)),

        layers.MaxPooling1D(pool_size=params['maxp2'], strides=None, padding="valid",),

        layers.Flatten(),

        layers.Dense(params['dense_neur'], activation='relu', ),

        layers.Dense(num_classes, activation='softmax',),

    ]) #

    lr = 0.001

    m.compile(optimizer=keras.optimizers.Adam(lr),

                  loss=keras.losses.categorical_crossentropy,

                  metrics=['accuracy'])

    return m
##

## Grid Search Params

## 

all_paramns = []

conv1_filters = reversed([64,32,16,8,4])

conv2_filters = reversed([64,32,16,8,4])

conv1_kernel = [5,3]

conv2_kernel = [5,3]

dense_neur = [64,32,16,4]

maxp1 = [4,2]

maxp2 = [4,2]

for a in conv1_filters:

    for b in conv2_filters:

        for c in conv1_kernel:

            for d in conv2_kernel:

                for e in dense_neur:

                    for f in maxp1:

                        for g in maxp2:

                            all_paramns.append({'conv1_filters':a,'conv2_filters':b,'conv1_kernel':c,

                                                     'conv2_kernel':d,'dense_neur':e,'maxp1':f,'maxp2':g})



print('Search in', len(all_paramns),' paramns')

print(all_paramns[0])
##

## Grid Search

## 

final_results = []

limit_s = 6

for param in all_paramns[:limit_s]:

    try:

        model = create_model_from_params(param)

        print('----------------------------')

        print(param)

        es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=100,

                          restore_best_weights=False)

        historia = model.fit(train_x_conv1d, train_y_onehot, 

                              epochs=200, verbose=0, batch_size=len(train_x), validation_split=0.2,

                             callbacks=[es])

        #

        

        acc = '{:.4}'.format(100*historia.history['accuracy'][-1])

        acc_val = '{:.4}'.format(100*historia.history['val_accuracy'][-1])

        print('Train acc:',acc)

        print('Val acc:',acc_val)

        final_results.append({'paramns':param_atual, 'acc':acc,'acc_val':acc_val})

        print('----------------------------')

    except:

        pass

        #print('Topology Error')

    