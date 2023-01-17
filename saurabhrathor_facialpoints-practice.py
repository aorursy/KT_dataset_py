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
print('Loading training ....')
data = pd.read_csv('../input/training.csv', error_bad_lines=False)
data.head()
images = data['Image'].str.split().as_matrix()
#images = data['Image']
images.shape
#images.shape
#data.shape
from math import sqrt
sqrt(len(images[0]))
print(len(images[0]))
SEQ_LEN = len(images[0])
print(images.shape)
#input_arr = np.array(7049, 9216)
input_arr = np.zeros((7049, 9216))
x=0
for i in images:
    #ln_arr = np.array((9216))
    y=0
    for j in i:
        input_arr[x,y] = j
        y=y+1
    x=x+1
    #input_arr = np.append(input_arr, ln_arr)
input_arr[:3,:20]
#input_arr = np.reshape(input_arr,(96,96))
print(input_arr.shape)
print(input_arr[2])

# extracting output data
output = data.loc[:, data.columns != 'Image']

#converting to numpy matrix
numpy_out = output.as_matrix()
print(numpy_out[:2])
print(numpy_out.shape)
# min-max normalization of output data to compress it in range 0-1
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
numpy_out_scaled = min_max_scaler.fit_transform(numpy_out)
print(numpy_out_scaled[:1])

input_arr_scaled = min_max_scaler.fit_transform(input_arr)
print(input_arr_scaled[:1])
# reshaping in image format
input_array = input_arr_scaled.reshape((7049,96,96,1))
print(input_array.shape)
print(input_array[2,-1,:,0])
# reshaping and splitting in trainging and validation set
input_train = input_array[:6000, :, :, :]
input_val = input_array[6000:, :, :, :]
out_train = numpy_out_scaled[:6000,:]
out_val = numpy_out_scaled[6000:, :]
# Importing keras libraries
from keras.layers import Dense, Input, GlobalMaxPooling1D, Flatten
from keras.layers import Conv1D, MaxPooling2D, Embedding, UpSampling1D, Reshape, Conv2D, Dropout
from keras.models import Model
from keras.models import model_from_json
from keras import optimizers
# designing Neural Net
print('Building model...')

# train a 1D convnet
input_ = Input(shape=(96, 96, 1))
print('input', input_.shape)
#x = Reshape((96,96,1))(input_)
x = Conv2D(nb_filter=100, kernel_size=(4,4), activation='sigmoid' )(input_)
print('Conv1', x.shape)
#x = Dropout(0.4)(x)
x = MaxPooling2D(pool_size=(2,2), padding='valid')(x)
print('Maxpool1', x.shape)

#### Layer2 ####
x = Conv2D(nb_filter=70, kernel_size=(4,4), activation='sigmoid' )(x)
print('Conv2', x.shape)
#x = Dropout(0.4)(x)
x = MaxPooling2D(pool_size=(4,4), padding='valid')(x)
print('Maxpool2', x.shape)

#### Layer3 ####
x = Conv2D(nb_filter=30, kernel_size=(4,4), activation='sigmoid' )(x)
print('Conv3', x.shape)
#x = Dropout(0.4)(x)
x = MaxPooling2D(pool_size=(2,2), padding='valid')(x)
print('Maxpool3', x.shape)

## Flatten ###
x = Flatten()(x)
print('Flat', x.shape)

x = Dense(30, input_shape=(120,), activation='sigmoid')(x)
print('Dense1', x.shape)
x = Dense(30, input_shape=(60,), activation='sigmoid')(x)
print('Dense2', x.shape)
output = Dense(30, input_shape=(30,), activation='sigmoid')(x)
print('Dense3', output.shape)


model = Model(input_, output)
model.summary()
#rms = optimizers.RMSprop(lr=0.00001)
#model.compile(optimizer='RMSprop', loss='mse',metrics=['acc'])
model.compile(optimizer='adam', loss='mse',metrics=['acc'])
print(input_array.shape)
print(numpy_out.shape)
print(np.any(np.isnan(input_arr)))
batch_size = 10
epochs = 1
model.fit(input_train, out_train,
          epochs=epochs,
          batch_size=batch_size,
          verbose=1,
        validation_data=(input_val, out_val))
from keras import backend as K

# with a Sequential model
get_final_layer_output = K.function([model.layers[0].input], [model.layers[10].output])
layer_output = get_final_layer_output([input_train[:5]])[0]
