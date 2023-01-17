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
import pandas as pd
import cv2
import numpy as np
count = 0
df1 = pd.read_csv('/kaggle/input/fall-detection/fall_detection.csv')
for i in range(len(df1['Path'])):
    df1["Path"][i] = df1['Path'][i].replace("\\","/")
    df1["Path"][i] = df1['Path'][i].replace("frames/","/kaggle/input/frames/")
df1["Path"]
df1.loc[df1.Label == 'noFall', 'Label'] = 0
df1.loc[df1.Label == 'Fall', 'Label'] = 1
df = df1[:2000]
df2 = df1[2000:2500]
x = np.array([np.array(cv2.imread(fname)) for fname in df['Path']])
x_val = np.array([np.array(cv2.imread(fname)) for fname in df2['Path']])
x.shape,x_val.shape
# df.loc[df.Label == 'noFall', 'Label'] = 0
# df.loc[df.Label == 'Fall', 'Label'] = 1
y = np.array([np.array(fname) for fname in df['Label']])
y_val = np.array([np.array(fname) for fname in df2['Label']])
y = np.reshape(y,(2000,1))
y_val = np.reshape(y_val, (500,1))
y.shape,y_val.shape
x = np.reshape(x,(2000,480,720,3,1))
x_val = np.reshape(x_val,(500,480,720,3,1))
from keras.models import Sequential
import pandas as pd
import numpy as np
import datetime
import os
import sys
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import h5py

from keras.models import Model
from keras.layers import Input, Dense, MaxPooling2D, MaxPooling3D, Dropout, BatchNormalization, Flatten, Conv2D, Conv3D, AveragePooling2D, LSTM, Reshape
from keras import backend as K
from keras.callbacks import History 
from keras.layers import TimeDistributed
from keras.layers import ConvLSTM2D



model = Sequential()
# model.add(AveragePooling3D(pool_size=(1,4, 4),
#                    input_shape=(32,480, 720, 3),
#                    padding='same'))
# model.add(BatchNormalization())

model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),input_shape=(32,480, 720, 3),
                   padding='same', return_sequences=True))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),
                   padding='same', return_sequences=False))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(MaxPooling2D(pool_size=(4, 4), padding='same'))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Dense(1, activation='linear'))

print(model.summary())


model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mse','accuracy'])
def batch_generator(X, Y, batch_size = 32):
    indices = np.arange(len(X)) 
    print(indices)
    batch=[]
    while True:
            # it might be a good idea to shuffle your data before each epoch
#             np.random.shuffle(indices) 
            for i in indices:
                batch.append(i)
                if len(batch)==batch_size:
                    print(len(batch))
                    X[batch] = np.reshape(X[batch],(-1,batch_size, 480, 720, 3))
                    print(X[batch].shape)
                    Y[batch] = np.reshape(Y[batch], (-1,batch_size))
                    batch = []
                    yield X[batch], Y[batch]
train_batch= batch_generator(x, y, batch_size = 32)
model.fit_generator(train_batch, epochs=30, verbose=2,steps_per_epoch = 64)
def batch_size(x,y,batch_size,count):
    i = batch_size*count
    n = x.shape[0]
#     print(batch_size*(count+2))
    if batch_size*(count+2) <= n:
        temp_x = x[i:i+batch_size]
        temp_y = y[i:i+batch_size]
        temp_x = np.reshape(temp_x, (batch_size,480,720,3))
        temp_y = np.reshape(temp_y, (batch_size))
        return temp_x, temp_y
    temp_x = x[i:n]
    temp_y = y[i:n]
    SIZE = n-i
    temp_x = np.reshape(temp_x, (SIZE,480,720,3))
    temp_y = np.reshape(temp_y, (SIZE))
    return temp_x,temp_y
X_train, y_train = batch_size(x,y,32,1)
x_val, y_val = batch_size(x_val,y_val,8,1)
X_train.shape
for i in range(0,65):
    X_train, y_train = batch_size(x,y,32,i)
    x_val, y_val = batch_size(x_val,y_val,7,i)
    model.fit(X_train,y_train, validation_data=(x_val, y_val))
