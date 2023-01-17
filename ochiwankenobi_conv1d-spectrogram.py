# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

np.random.seed(42) # Set seed for random

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import time

import datetime

pd.options.display.precision = 15

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))



from tqdm import tqdm_notebook



from sklearn.model_selection import train_test_split



import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Dropout

from tensorflow.keras.layers import Conv1D, MaxPooling1D

from tensorflow.keras.layers import GlobalAveragePooling1D

from tensorflow.keras.layers import BatchNormalization, Activation

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from tensorflow.keras.initializers import he_normal



import warnings

warnings.filterwarnings("ignore")



from scipy.signal import spectrogram



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})

print("Train Data Loaded")
# Split into features and targets

X_train = train['acoustic_data'].values

y_train = train['time_to_failure'].values

# Clear the old stuff

del train
plt.hist(y_train,bins=50,facecolor='blue',density=True)

plt.title('Histogram of all time_to_failure values')

plt.show()
#rows = 150_000

#X_train = X_train[:int(np.floor(X_train.shape[0] / rows))*rows]

#y_train = y_train[:int(np.floor(y_train.shape[0] / rows))*rows]

#X_train= X_train.reshape((-1, rows, 1))

#y_train = y_train[rows-1::rows]

#train_starts = np.arange(rows,rows*4194,rows)



# Look at histogram of y_train values

#n, bins, patches = plt.hist(y_train,bins=50,facecolor='blue',density=True)

#plt.title('Histogram of y_train values')

#plt.show()
# First quartile, y_train < 4.0

Q1_indices = np.where(y_train<6.0)[0]

Q1_idx = Q1_indices > 150000

Q1_indices = Q1_indices[Q1_idx]

Q1_starts = np.random.choice(a=Q1_indices,size=4000)

del Q1_indices, Q1_idx

# Second Quartile, y_train > 4 && y_train < 8

Q2_indicesX = np.where(y_train>=4.0)[0]

Q2_indices1 = y_train >= 4.0

Q2_times = y_train[Q2_indices1]

del Q2_indices1     # free up memory asap

Q2_indices2 = Q2_times < 10.0

Q2_indices = Q2_indicesX[Q2_indices2]    # Limit between 3 and 9

del Q2_indices2, Q2_indicesX     # free up memory asap

idx = Q2_indices > 150000

Q2_indices = Q2_indices[idx]

Q2_starts = np.random.choice(a=Q2_indices,size=3000)

del Q2_indices, Q2_times, idx

# Third quartile, y_train >= 7

Q3_indices = np.where(y_train>=4.0)[0]

Q3_idx = Q3_indices > 150000

Q3_indices = Q3_indices[Q3_idx]

Q3_starts = np.random.choice(a=Q3_indices,size=4000)

del Q3_idx, Q3_indices



# Concatenate the starts arrays

train_starts = np.r_[Q1_starts, Q2_starts, Q3_starts]

del Q1_starts, Q2_starts, Q3_starts



# Get y_tr

y_train = y_train[train_starts]



# Look at histogram of y_tr values

plt.hist(y_train,bins=50,facecolor='blue',density=True)

plt.title('Histogram of y_tr values')

plt.show()
# Gxx params

NPERSEG = 4

NOVERLAP = 0

Gxx_train = np.zeros([len(y_train),37500,2])

# Choose a random length between 100k and 300k

sig_length = 150000

for ii, seg in tqdm_notebook(enumerate(train_starts)):

    # Grab the chunk of signal

    x_time = X_train[train_starts[ii]-sig_length:train_starts[ii]]

    # Spectrogram the signal

    _, __, Gxx = spectrogram(x_time,fs=1.0,window='hann', noverlap=NOVERLAP, 

                             return_onesided=True, nperseg=NPERSEG, 

                             scaling='density', mode='magnitude')

    # Remove all frequencies above indice 2

    Gxx = Gxx[0:2,:]

    # Reshape Gxx as a 3D array

    Gxx = Gxx.reshape([1,Gxx.shape[1],Gxx.shape[0]])

    # Allocate into main 3D array

    Gxx_train[ii,:,:] = Gxx[0,:,:]

        

        

# Clear memory from train data

del X_train, x_time, sig_length
# Split train and validation sets

indices = np.arange(0,Gxx_train.shape[0],1)

train_indices, val_indices = train_test_split(indices,test_size=0.2,random_state=42)

del indices

# Split up with indices

Gxx_val = Gxx_train[val_indices,:,:]

Gxx_train = Gxx_train[train_indices,:,:]

y_val = y_train[val_indices]

y_train = y_train[train_indices]
# Lets build a CNN model

print('Building CNN Model 1') # Diagnostics purposes

input_shape = (Gxx.shape[1],Gxx.shape[2])

batch_size = 64

decay = 0.0001 / 100

kernel_initializer = tf.keras.initializers.RandomUniform(seed=1)

# Conv2D model

with tf.device('/gpu:0'):

    mdl1 = tf.keras.models.Sequential()

    mdl1.add(Conv1D(filters=16,kernel_size=10, padding="same", input_shape=input_shape,

                    kernel_initializer=kernel_initializer, activation='relu'))

    mdl1.add(Conv1D(filters=16,kernel_size=10, padding="same", kernel_initializer=kernel_initializer, activation='relu'))

    mdl1.add(MaxPooling1D(pool_size=100))

    mdl1.add(Conv1D(filters=32,kernel_size=10,padding="same", kernel_initializer=kernel_initializer, activation='relu'))

    mdl1.add(Conv1D(filters=32,kernel_size=10,padding="same", kernel_initializer=kernel_initializer, activation='relu'))

    mdl1.add(GlobalAveragePooling1D())

    mdl1.add(Dense(16, kernel_initializer=kernel_initializer, activation='relu'))

    mdl1.add(Dense(1,activation='linear'))

    # Early Stopping and stuff

    earlyStopping = EarlyStopping(monitor='val_loss',

                              patience=10,

                              verbose=1,

                              mode='min',

                              )

    mcp_save = ModelCheckpoint('.mdl1_wts.hdf5',

                           save_best_only=True,

                           monitor='val_loss',

                           mode='min')

    

    # Compile the model

    mdl1.compile(loss=tf.keras.losses.mean_absolute_error,

                optimizer=tf.keras.optimizers.SGD(0.01,momentum=0.7,decay=decay),

                metrics=['mae'])





    t1 = time.time()

    mdl1.fit(Gxx_train,y_train,

            batch_size=batch_size,

            epochs=100,

            verbose=1,

            validation_data= (Gxx_val, y_val),

            callbacks=[earlyStopping, mcp_save]

            )

    t_total = time.time() - t1

    print("Time for train: ",str(t_total/60**2)," hours")



# CNN model with different initialization

print('Building CNN Model 2') # Diagnostics purposes

input_shape = (Gxx.shape[1],Gxx.shape[2])

batch_size = 64

kernel_initializer = tf.keras.initializers.RandomNormal(seed=11)

# Conv2D model

with tf.device('/gpu:0'):

    mdl2 = tf.keras.models.Sequential()

    mdl2.add(Conv1D(filters=16,kernel_size=10, padding="same", input_shape=input_shape,

                    kernel_initializer=kernel_initializer, activation='relu'))

    mdl2.add(Conv1D(filters=16,kernel_size=10, padding="same", kernel_initializer=kernel_initializer, activation='relu'))

    mdl2.add(MaxPooling1D(pool_size=100))

    mdl2.add(Conv1D(filters=32,kernel_size=10,padding="same",kernel_initializer=kernel_initializer, activation='relu'))

    mdl2.add(Conv1D(filters=32,kernel_size=10,padding="same",kernel_initializer=kernel_initializer, activation='relu'))

    mdl2.add(GlobalAveragePooling1D())

    mdl2.add(Dense(16,activation='relu',kernel_initializer=kernel_initializer))

    mdl2.add(Dense(1,activation='linear'))

    # Early Stopping and stuff

    earlyStopping = EarlyStopping(monitor='val_loss',

                              patience=10,

                              verbose=1,

                              mode='min',

                              )

    mcp_save = ModelCheckpoint('.mdl2_wts.hdf5',

                           save_best_only=True,

                           monitor='val_loss',

                           mode='min')

    

    # Compile the model

    mdl2.compile(loss=tf.keras.losses.mean_absolute_error,

                optimizer=tf.keras.optimizers.SGD(0.01,momentum=0.7,decay=decay),

                metrics=['mae'])





    t1 = time.time()

    mdl2.fit(Gxx_train,y_train,

            batch_size=batch_size,

            epochs=100,

            verbose=1,

            validation_data= (Gxx_val, y_val),

            callbacks=[earlyStopping, mcp_save]

            )

    t_total = time.time() - t1

    print("Time for train: ",str(t_total/60**2)," hours")





# Simple CNN model

print('Building CNN Model 3') # Diagnostics purposes

input_shape = (Gxx.shape[1],Gxx.shape[2])

batch_size = 64

kernel_initializer = tf.keras.initializers.RandomNormal(seed=12)

# Conv2D model

with tf.device('/gpu:0'):

    mdl3 = tf.keras.models.Sequential()

    mdl3.add(Conv1D(filters=16,kernel_size=10, padding="same", input_shape=input_shape,

                    kernel_initializer=kernel_initializer, activation='relu'))

    mdl3.add(Conv1D(filters=16,kernel_size=10, padding="same", 

                    kernel_initializer=kernel_initializer, activation='relu'))

    mdl3.add(MaxPooling1D(pool_size=100, strides=1))

    mdl3.add(Conv1D(filters=32,kernel_size=10,padding="same",

                    kernel_initializer=kernel_initializer, activation='relu'))

    mdl3.add(Conv1D(filters=32,kernel_size=10,padding="same",

                    kernel_initializer=kernel_initializer, activation='relu'))

    mdl3.add(GlobalAveragePooling1D())

    mdl3.add(Dense(16,activation='relu',kernel_initializer=kernel_initializer))

    mdl3.add(Dense(1,activation='linear'))

    # Early Stopping and stuff

    earlyStopping = EarlyStopping(monitor='val_loss',

                              patience=10,

                              verbose=1,

                              mode='min',

                              )

    mcp_save = ModelCheckpoint('.mdl3_wts.hdf5',

                           save_best_only=True,

                           monitor='val_loss',

                           mode='min')

    

    # Compile the model

    mdl3.compile(loss=tf.keras.losses.mean_absolute_error,

                optimizer=tf.keras.optimizers.SGD(0.01,momentum=0.7,decay=decay),

                metrics=['mae'])





    t1 = time.time()

    mdl3.fit(Gxx_train,y_train,

            batch_size=batch_size,

            epochs=100,

            verbose=1,

            validation_data= (Gxx_val, y_val),

            callbacks=[earlyStopping, mcp_save]

            )

    t_total = time.time() - t1

    print("Time for train: ",str(t_total/60**2)," hours")



# Clear Gxx_train, y_tr

del Gxx_train, y_train
# Format test data

submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')

# Convert X_test to Gxx

Gxx_test = np.zeros([len(submission),37500,2])

for ii, seg in tqdm_notebook(enumerate(submission.index)):

    # Grab the chunk of signal

    x_time = pd.read_csv('../input/test/' + seg + '.csv').values.astype(np.int16)

    # Spectrogram the signal

    _, __, Gxx = spectrogram(x_time.reshape(-1),fs=1.0,window='hann', nperseg=NPERSEG, 

                             noverlap=NOVERLAP, return_onesided=True, 

                             scaling='density', mode='magnitude')

    # Remove information for freqs above indice 32

    Gxx = Gxx[0:2,:]

    # Reshape Gxx as a 3D array

    Gxx = Gxx.reshape([1,Gxx.shape[1],Gxx.shape[0]])

    # Allocate into main 3D array

    Gxx_test[ii,:,:] = Gxx[0,:,:]      
# Get predictions for each test instance

t1 = time.time()

with tf.device('/gpu:0'):

    mdl1.load_weights(".mdl1_wts.hdf5")   # Restore best weights

    mdl2.load_weights(".mdl2_wts.hdf5")

    mdl3.load_weights(".mdl3_wts.hdf5")

    # Evaluate on validation set

    loss1, mae_best1 = mdl1.evaluate(Gxx_val,y_val)

    loss2, mae_best2 = mdl2.evaluate(Gxx_val,y_val)

    loss3, mae_best3 = mdl3.evaluate(Gxx_val,y_val)

    # Get predictions from models to look at distributions later

    mdl1_val_preds = mdl1.predict(Gxx_val,batch_size=32)

    mdl2_val_preds = mdl2.predict(Gxx_val,batch_size=32)

    mdl3_val_preds = mdl3.predict(Gxx_val,batch_size=32)

    # Print what the best validation MAE was for each model

    print('Mdl1 Best Validation MAE: ',str(mae_best1))

    print('Mdl2 Best Validation MAE: ',str(mae_best2))

    print('Mdl3 Best Validation MAE: ',str(mae_best3))

    # Get predictions on the test set for each model

    mdl1_test_preds = mdl1.predict(Gxx_test,batch_size=32)

    mdl2_test_preds = mdl2.predict(Gxx_test,batch_size=32)

    mdl3_test_preds = mdl3.predict(Gxx_test,batch_size=32)

t_total = time.time() - t1

print("Time for test predictions: ",str(t_total/60)," minutes")



# Prepare the submission

val_preds = (mdl1_val_preds + mdl2_val_preds + mdl3_val_preds) / 3

test_preds = (mdl1_test_preds + mdl2_test_preds + mdl3_test_preds) / 3 # Blending CNN models

submission['time_to_failure'] = test_preds



# Convert the submission to .csv

submission.to_csv('submission.csv')



# Look at validation distributions

BINS = np.linspace(0,16,100)

plt.figure(figsize=(6.0,3.0),dpi=150)

plt.hist(mdl1_val_preds, bins=BINS, density=True, label='Mdl1',alpha=0.2)

plt.hist(mdl2_val_preds, bins=BINS, density=True, label='Mdl2',alpha=0.2)

plt.hist(mdl3_val_preds, bins=BINS, density=True, label='Mdl3',alpha=0.2)

plt.hist(val_preds, bins=BINS, density=True, label='Blend',alpha=0.3)

plt.title('Validation Predictions Distribution')

plt.legend(loc='upper right')
# Look at the submission distributions

BINS2 = np.linspace(0,16,100)

plt.figure(figsize=(6.0,3.0),dpi=150)

plt.hist(mdl1_test_preds, bins=BINS2, density=True, label='Mdl1', alpha=0.4)

plt.hist(mdl2_test_preds, bins=BINS2, density=True, label='Mdl2', alpha=0.4)

plt.hist(mdl3_test_preds, bins=BINS2, density=True, label='Mdl3', alpha=0.4)

plt.hist(test_preds,bins=BINS2,density=True,label='Blend', alpha=0.5)

plt.title('Test Predictions Distributions')

plt.legend(loc='upper right')

plt.show()