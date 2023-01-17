# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import random

import math



from kaggle_datasets import KaggleDatasets

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from tensorflow.keras import backend

from tensorflow.keras.models import Sequential, Model, load_model

from tensorflow.keras.layers import Input, Add, Dense, Activation, BatchNormalization, Flatten, AveragePooling2D

from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, GRU, LSTM, Concatenate, Bidirectional, GlobalAveragePooling1D

from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, LearningRateScheduler

from tensorflow.keras.optimizers import SGD, Adam

from tensorflow.keras.initializers import glorot_uniform

from tensorflow.keras import regularizers

import keras.backend as K



import tensorflow as tf





### Plot model ###

from tensorflow.keras.utils import plot_model



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Detect hardware, return appropriate distribution strategy

try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.



print("REPLICAS: ", strategy.num_replicas_in_sync)


SMALL_SIZE = 8

MEDIUM_SIZE = 12

BIGGER_SIZE = 24

xyaxislabel = 6



plt.rc('font', size=SMALL_SIZE)          # controls default text sizes

plt.rc('axes', titlesize=xyaxislabel)     # fontsize of the axes title

plt.rc('axes', labelsize=xyaxislabel)    # fontsize of the x and y labels

plt.rc('xtick', labelsize=xyaxislabel)    # fontsize of the tick labels

plt.rc('ytick', labelsize=xyaxislabel)    # fontsize of the tick labels

plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize

plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# %matplotlib qt
try:

    GCS_PATH = KaggleDatasets().get_gcs_path() # you can list the bucket with "!gsutil ls $GCS_PATH"

except:

    pass
import pandas as pd

sample_submission = pd.read_csv("../input/liverpool-ion-switching/sample_submission.csv")

test = pd.read_csv("../input/liverpool-ion-switching/test.csv", sep = ',', dtype = {'Time': float, 'Signal':float})

data = pd.read_csv("../input/liverpool-ion-switching/train.csv", sep = ',', dtype = {'Time':float, 'Signal':float, 'Open_channels':int})
data.index = ((data.time * 10_000) - 1).values

test.index = ((test.time * 10_000) - 1).values

data.index = data.index//500_000

test.index = test.index//500_000



std_mult = 100



## Not using Outlier limits ##

for idx in data.index.unique():

    batch_mean = data.loc[idx].signal.mean()

    batch_std = data.loc[idx].signal.std()

    outlier_limit_high = batch_std * std_mult

    outlier_limit_low = batch_std * -1 * std_mult

    batch_median = data.loc[idx].signal.median()

    data.loc[idx].signal = (data.loc[idx].signal - batch_mean) / batch_std

#     data.loc[idx].signal = np.where(data.loc[idx].signal > outlier_limit_high, batch_median, data.loc[idx].signal)

#     data.loc[idx].signal = np.where(data.loc[idx].signal < outlier_limit_low, batch_median, data.loc[idx].signal)



for idx in test.index.unique():

    batch_mean = test.loc[idx].signal.mean()

    batch_std = test.loc[idx].signal.std()

    outlier_limit_high = batch_std * std_mult

    outlier_limit_low = batch_std * -1 * std_mult

    batch_median = test.loc[idx].signal.median()

    test.loc[idx].signal = (test.loc[idx].signal - batch_mean) / batch_std

#     test.loc[idx].signal = np.where(test.loc[idx].signal > outlier_limit_high, batch_median, test.loc[idx].signal)

#     test.loc[idx].signal = np.where(test.loc[idx].signal < outlier_limit_low, batch_median, test.loc[idx].signal)


# f, axis = plt.subplots(5,2)

# for index, ax in enumerate(axis.flat):

#     ax.plot(data.loc[float(index)].signal.values)

# #     ax.plot(data.loc[float(index)].open_channels.values)

#     ax.set(title = str(index*50)+' seconds to '+str((index+1)*50))

#     ax.label_outer()


# f, axis = plt.subplots(2,2)

# for index, ax in enumerate(axis.flat):

#     ax.plot(test.loc[float(10+index)].signal.values)

#     ax.set(title = str(10+index*50)+' seconds to '+str((index+11)*50))

#     ax.label_outer()
X = data.signal.values.reshape(-1,1000,1)

y = pd.get_dummies(data.open_channels).values.reshape(-1, 1000, 11)



ind_list = [i for i in range(X.shape[0])]

random.seed(200)

random.shuffle(ind_list)

X_shuffled  = X[ind_list, :,:]

y_shuffled = y[ind_list,]





split_1 = int(0.8 * len(X_shuffled))

split_2 = int(1 * len(X_shuffled))

X_train = X_shuffled[:split_1].astype(np.float32)

X_dev = X_shuffled[split_1:split_2].astype(np.float32)



y_train = y_shuffled[:split_1].astype(np.int32)

y_dev = y_shuffled[split_1:split_2].astype(np.int32)

X_train.dtype
y_train.dtype


def recall_m(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (possible_positives + K.epsilon())

    return recall



def precision_m(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())

    return precision



def f1_m(y_true, y_pred):

    precision = precision_m(y_true, y_pred)

    recall = recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))



def step_decay(epoch):

    initial_lrate = 0.01

    drop = 0.5

    epochs_drop = 10

    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))

    return lrate
def conv_concat_layer(x, filters, kernel_size, stride):

    

    x1 = Conv1D(filters=filters, kernel_size = kernel_size, strides = stride, padding = 'same', kernel_initializer = glorot_uniform())(x)

    x1 = BatchNormalization()(x1)

    x1 = Activation('relu')(x1)

    

    x2 = Conv1D(filters=filters, kernel_size = kernel_size*2, strides = stride, padding = 'same', kernel_initializer = glorot_uniform())(x)

    x2 = BatchNormalization()(x2)

    x2 = Activation('relu')(x2)

    

    x3 = Conv1D(filters=filters, kernel_size = kernel_size*3, strides = stride, padding = 'same', kernel_initializer = glorot_uniform())(x)

    x3 = BatchNormalization()(x3)

    x3 = Activation('relu')(x3)

    

    x = Add()([x1, x2, x3])

    

    x = Dense(1000, activation = 'relu', kernel_initializer = glorot_uniform())(x)

    

    return x
def GRU_concat_layer(x, n_units):

    x = Bidirectional(GRU(n_units, return_sequences = True))(x)

    X1 = x

    x = Bidirectional(GRU(n_units, return_sequences = True))(x)

    X2 = x

    x = Add()([x, X1])

    

    x = Bidirectional(GRU(n_units, return_sequences = True))(x)

    x = Add()([x, X1, X2])

    

    return x

    
def chrononet(input_shape, filters, kernel_size, stride, n_units):

    

    input_layer = Input(input_shape)

    x = conv_concat_layer(input_layer , filters, kernel_size, stride)

    x = conv_concat_layer(x, filters, kernel_size, stride)

    x = conv_concat_layer(x, filters, kernel_size, stride)

#     x = conv_concat_layer(x, filters, kernel_size, stride)

#     x = conv_concat_layer(x, filters, kernel_size, stride)

    

    ## GRU Shape needs some research ## 

    x = GRU_concat_layer(x, n_units)

#     x = GRU_concat_layer(x, n_units = 32)



    ## Out layer ##

    x = Conv1D(11, kernel_size = kernel_size, strides = 1, padding = 'same')(x)

    out = Activation('softmax')(x)

    

    model = Model(input_layer, out)

    

    

    return model
def Chrononet():



################################################

################################################

    input_shape = (None, 1)

    filters = 64 ## Filter size

    kernel_size = 2 ## Kernel size

    stride = 1 ## Strides

    n_units = 32 ## Control number of GRU units

################################################    

################################################    

    with strategy.scope():

        model = chrononet(input_shape, filters, kernel_size, stride, n_units)

        model.compile(loss = 'categorical_crossentropy', 

                  optimizer = Adam(lr = 5e-5),

                  metrics = ['acc', f1_m])



    return model
model = Chrononet()
plot_model(model, show_shapes=True, show_layer_names=True)
lrate = LearningRateScheduler(step_decay)



### Stop early if val loss goes out of control, wait 200 epochs before stopping ###

es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 0, patience = 20)

### Save best model out of run ###

#     mc = ModelCheckpoint(filepath = '/best_chrononet_model.h5', 

#                          monitor = 'val_acc', mode = 'max', verbose = 0, save_best_only = True)     



history = model.fit(X_train, y_train, 

                epochs = 64, 

                batch_size = 32 ,

                verbose = 1,

                validation_data = (X_dev, y_dev), 

                callbacks = [es, lrate],

                )
loss, accuracy, f1m = model.evaluate(X_train, y_train, verbose=1)

loss_dev, accuracy_dev, f1m_dev = model.evaluate(X_dev, y_dev, verbose=1)
model.summary()
plt.figure(figsize = (16,12), dpi = 80)

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Accuracy vs epochs', fontsize = 20)

plt.ylabel('Accuracy', fontsize = 16)

plt.xlabel('epoch', fontsize = 16)

plt.legend(['Training Accuracy','Validation Accuracy'], loc='lower left')

plt.ylim(0,1)
plt.figure(figsize = (16,12), dpi = 80)

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Loss vs epochs', fontsize = 20)

plt.ylabel('Loss', fontsize = 16)

plt.xlabel('epoch', fontsize = 16)

plt.legend(['Training Loss','Validation Loss'], loc='upper left')

plt.ylim(0,5)
sub = pd.read_csv('../input/liverpool-ion-switching/sample_submission.csv', dtype = dict(time = str))

X_test = test.signal.values.reshape(-1, 1000, 1)
test_pred = model.predict(X_test, batch_size=64).argmax(axis=-1)

sub.open_channels = test_pred.reshape(-1)

sub.to_csv('submission.csv', index=False)
model.save_weights('model_rough.h5')