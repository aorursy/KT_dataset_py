# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.preprocessing import StandardScaler

from sklearn.svm import NuSVR

from sklearn.metrics import mean_absolute_error
train = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.float32, 'time_to_failure': np.float32}).values
train [0:150000, 0 ] .mean(axis=0)
# pandas doesn't show us all the decimals

pd.options.display.precision = 15
rows = 150_000

segments = int(np.floor(train.shape[0] / rows))

print('train.shape',train.shape)

segments

def create_X(x, last_index=None, n_steps=150, step_length=1000):

    if last_index == None:

        last_index=len(x)

       

    assert last_index - n_steps * step_length >= 0



    

    # Reshaping and approximate standardization with mean 5 and std 3.

    temp = (x[(int(last_index) - n_steps * step_length):int(last_index)].reshape(n_steps,step_length ).astype(np.float32) - 5 ) / 3   

    # convert (150000) to [150 1000 ]

    # then extract feature from each row of length 1000. so total 150 

    

    # Extracts features of sequences of full length 1000, of the last 100 values and finally also 

    # of the last 10 observations. 

    

    return temp

# Query "create_X" to figure out the number of features

n_features = create_X(train [0:150000,0]).shape

print("Output segment shape", n_features)     # 18 features each row of segment ie 150x18 features of 150000 chunk input



n_steps=150

step_length=1000

maxsize=train .shape[0]

seg = int(np.floor(maxsize / (n_steps*step_length))) 

batch_size = seg-1   # (4193,) 

xx=350





##############################################################################################

rows_initialize = np.zeros((seg), dtype=float)

print(rows_initialize.shape)



for seg1 in tqdm(range(1,seg)) :      # for loop from 1 to 4194 segment value

    rows_initialize [seg1] = seg1 * (n_steps*step_length) 



rows=np.delete(rows_initialize,0)    # (4193,)



print(rows.shape)



########################################################################################

batch_size=batch_size-xx    # training data

#batch_size=xx              # validation data

split_point=xx

second_earthquake = rows[xx]







##########################################################################################



if batch_size < 1000  :    # validation set 

               rows_1 = rows[:split_point+1]    #  0:350 

        

if batch_size > 1000 :   # training set

               rows_1 = rows[split_point+1 :]    # (351,) ie 351:4193    

            



       

    # Initialize feature matrices and targets

samples_tr= np.zeros((rows_1.shape[0], n_features[0], n_features[1]), dtype=float)   #  for validation (350,150000)  for training ( 3842, 150000) 

targets_tr = np.zeros(rows_1.shape[0], )    # (16,)  for validation (350)    for training ( 3843)

        

for j, row in enumerate(rows_1):             # 16 for validation (350)    for training ( 3843)

    samples_tr[j] = create_X(train[:, 0], last_index=row, n_steps=n_steps, step_length=step_length)

    targets_tr[j] = train[int(row - 1), 1]         

    

    

################################################################################################



print('samples_tr shape', samples_tr.shape)

print('targets_tr shape', targets_tr.shape)



samples_tr.shape

#batch_size=batch_size-xx    # training data

batch_size=xx              # validation data

split_point=xx

second_earthquake = rows[xx]



##########################################################################################



if batch_size < 1000  :    # validation set 

               rows_1 = rows[:split_point+1]    #  0:350 

        

if batch_size > 1000 :   # training set

               rows_1 = rows[split_point+1 :]    # (351,) ie 351:4193    

            



       

    # Initialize feature matrices and targets

samples_vd= np.zeros((rows_1.shape[0], n_features[0], n_features[1]), dtype=float)    #  for validation (350,150000)  for training ( 3842, 150000) 

targets_vd = np.zeros(rows_1.shape[0], )    # (16,)  for validation (350)    for training ( 3843)

        

for j, row in enumerate(rows_1):             # 16 for validation (350)    for training ( 3843)

    samples_vd[j] = create_X(train[:, 0], last_index=row, n_steps=n_steps, step_length=step_length)

    targets_vd[j] = train[int(row - 1), 1]         

    

    

################################################################################################



    

print('samples_tr shape', samples_tr.shape)

print('targets_tr shape',targets_tr.shape) 

    

print('samples_vd shape', samples_vd.shape)

print('targets_vd shape',targets_vd.shape)  

#print('rows_1 shape',rows_1.shape[0])

    
from keras.models import Sequential

from keras.layers import Dense, CuDNNGRU, SimpleRNN, LSTM ,  Dropout, Activation, Flatten, Input, Conv1D, MaxPooling1D

from keras.optimizers import adam

from keras.callbacks import ModelCheckpoint

from keras.optimizers import RMSprop

## CNN combined with LSTM Model 

i = (n_features[0],n_features[1])

model = Sequential ()



model.add(Conv1D (kernel_size = (3), filters = 32, strides=2, input_shape=i, kernel_initializer='he_normal', activation='relu')) 

#model.add(BatchNormalization())

model.add(Dropout(0.2))



model.add(Conv1D (kernel_size = (3), filters = 16, strides=2, kernel_initializer='he_normal', activation='relu')) 

#model.add(BatchNormalization())

model.add(Dropout(0.2))

model.add(MaxPooling1D())



model.add(Conv1D (kernel_size = (3), filters = 8, strides=2, kernel_initializer='he_normal', activation='relu')) 

#model.add(BatchNormalization())

model.add(Dropout(0.2))

model.add(MaxPooling1D())





#model.add(Flatten())

#model.add(Dense (250, activation='relu', kernel_initializer='he_normal'))

#model.add(BatchNormalization())

#model.add(Dropout(0.5))

    

model.add(LSTM(256,  return_sequences=True))

model.add(Dropout(0.2))

model.add(LSTM(128))

model.add(Dropout(0.2))











model.add(Dense(256))

model.add(Dense(128))

model.add(Dense(64))

model.add(Dense(32))

model.add(Dense(16))

model.add(Dense(8))

model.add(Dense(4))

model.add(Dense(1))







##1st model

#model.add(Conv1D(5, 3, activation='relu', input_shape= i))

#model.add(MaxPooling1D(2))

#model.add(LSTM(50,  return_sequences=True))

#model.add(LSTM(10))

#model.add(Dense(240))

#model.add(Dense(120))

#model.add(Dense(60))

#model.add(Dense(30))

#model.add(Dense(1))



##2nd model 



#model.add(Conv1D(16, 3, activation='relu', input_shape= i))

#model.add(MaxPooling1D(2))

#model.add(Conv1D(128, 3, activation='relu'))

#model.add(MaxPooling1D(2))

#model.add(Conv1D(16, 3, activation='relu'))

#model.add(MaxPooling1D(2))

#model.add(Dropout(0.1))

#model.add(LSTM(48,dropout=0.2, recurrent_dropout=0.2, return_sequences=True))

#model.add(LSTM(20,dropout=0.2, recurrent_dropout=0.2, return_sequences=False))

#model.add(Dense(1, activation='linear'))

 
model.summary()
import keras

from keras.optimizers import RMSprop

opt = keras.optimizers.adam(lr=.005)



model.compile(loss="mae",

              optimizer=opt, metrics=['mean_absolute_error'])

             # metrics=['accuracy'])





batch_size = 128 # mini-batch with 32 examples

epochs = 50

history = model.fit(

    samples_tr, targets_tr,

    batch_size=batch_size,

    epochs=epochs,

    verbose=1)

   # validation_data=(samples_vd  ,targets_vd ))
submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id', dtype={"time_to_failure": np.float32})
for i, seg_id in enumerate(tqdm(submission.index)):

  #  print(i)

    seg = pd.read_csv('../input/test/' + seg_id + '.csv')

    x = seg['acoustic_data'].values

    submission.time_to_failure[i] = model.predict(np.expand_dims(create_X(x), 0))



submission.head()
submission.to_csv('submission.csv')