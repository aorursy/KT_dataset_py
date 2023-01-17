# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from keras import Sequential

from keras.layers import Conv2D, MaxPool2D, Dropout, Dense, Flatten, Reshape, Conv2DTranspose, UpSampling2D, Input

from keras.models import Model

from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
train      = '/kaggle/input/intel-image-classification/seg_train/seg_train'

validation = '/kaggle/input/intel-image-classification/seg_test/seg_test'

test       = '/kaggle/input/intel-image-classification/seg_pred'
no_of_images = 6000

target_size = 64
train_datagen = ImageDataGenerator(rescale = 1./255)



train_generator = train_datagen.flow_from_directory(train,

                                                   target_size = (target_size, target_size),

                                                   batch_size = no_of_images,

                                                   class_mode=None)
for img_batch in train_generator:

    plt.figure(figsize=(20,10))

    for ix in range(32):

        sub = plt.subplot(4, 8, ix + 1)

        plt.imshow(img_batch[ix])

        plt.xticks([])

        plt.yticks([])

    break

        
def build_simple_autoencoder(input_shape = 128, embedding_dim = 1048):

    total_feature = input_shape * input_shape * 3

    input_img = Input(shape=(total_feature, ))

    encoded = Dense(embedding_dim, activation = 'relu')(input_img)

    decoded = Dense(total_feature, activation = 'sigmoid')(encoded)

    

    model = Model(input_img, decoded)

    model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

    

    return model
def build_deep_autoencoder(input_shape = 128, embedding_dim = 1048):

    input_shape = (input_shape, input_shape, 3)

    input_img = Input(shape = input_shape)

    encoded   = Conv2D(64, kernel_size = (3, 3), strides = (2, 2), padding = 'same', activation = 'relu')(input_img)

    encoded   = Conv2D(128, kernel_size = (3, 3), strides = (2, 2), padding = 'same', activation = 'relu')(encoded)    

    encoded   = MaxPool2D()(encoded)

    encoded   = Conv2D(256, kernel_size = (3, 3), strides = (2, 2), padding = 'same', activation = 'relu')(encoded)

    encoded   = Conv2D(512, kernel_size = (3, 3), strides = (2, 2), padding = 'same', activation = 'relu')(encoded)    

    encoded   = Flatten()(encoded)

    encoded   = Dense(embedding_dim, activation = 'sigmoid')(encoded)

    

    decoded  = Dense(2 * 2 * 512, activation = 'sigmoid')(encoded)

    decoded  = Reshape((2, 2, 512))(decoded)

    decoded  = Conv2DTranspose(512, kernel_size = (3, 3), strides = (2, 2), padding = 'same', activation = 'relu')(decoded)

    decoded  = Conv2DTranspose(256, kernel_size = (3, 3), strides = (2, 2), padding = 'same', activation = 'relu')(decoded)   

    decoded  = UpSampling2D()(decoded)

    decoded  = Conv2DTranspose(128, kernel_size = (3, 3), strides = (2, 2), padding = 'same', activation = 'relu')(decoded)

    decoded  = Conv2DTranspose(64,  kernel_size = (3, 3), strides = (2, 2), padding = 'same', activation = 'relu')(decoded)      

    decoded  = Conv2DTranspose(3,   kernel_size = (2, 2), strides = (1, 1), padding = 'same', activation = 'sigmoid')(decoded)      

    

    

    model = Model(input_img, decoded)

    model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

    return model

    
def build_deep_autoencoder_2(input_shape = 128, embedding_dim = 1048):

    input_shape = (input_shape, input_shape, 3)

    

    model = Sequential()

    model.add(Conv2D(64, kernel_size = (3,3), padding = 'same', activation = 'relu', input_shape = input_shape))

    model.add(Conv2D(128, kernel_size = (3,3), padding = 'same', activation = 'relu'))

    model.add(MaxPool2D())

    model.add(Conv2D(256, kernel_size = (3, 3), padding = 'same',activation = 'relu'))

    model.add(Conv2D(256, kernel_size = (3, 3), padding = 'same',activation = 'relu'))

    model.add(MaxPool2D())

    model.add(Conv2D(512, kernel_size = (3, 3), padding = 'same',activation = 'relu')) 

    model.add(Conv2D(512, kernel_size = (3, 3), padding = 'same',activation = 'relu')) 

    model.add(Flatten())

    model.add(Dense(embedding_dim, activation = 'relu'))

    

    model.add(Dense(2 * 2 * 512, activation = 'relu'))

    model.add(Reshape((2 , 2 , 512)))    

    model.add(Conv2DTranspose(512, kernel_size = (3, 3), padding = 'same',activation = 'relu'))

    model.add(UpSampling2D())

    model.add(Conv2DTranspose(256, kernel_size = (3, 3), strides = (2, 2), padding = 'same',activation = 'relu'))

    model.add(Conv2DTranspose(256, kernel_size = (3, 3), strides = (2, 2), padding = 'same',activation = 'relu'))    

    model.add(UpSampling2D())    

    model.add(Conv2DTranspose(128, kernel_size = (3, 3), padding = 'same',activation = 'relu'))

    model.add(Conv2DTranspose(3,   kernel_size = (3, 3), strides = (2, 2), padding = 'same',activation = 'relu'))

    

    

    model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

    return model

    
def get_autoencoder_model(model_type = 'simple', target_size = 128, embedding_dim = 1048):

    if model_type == 'simple':

        model = build_simple_autoencoder(input_shape = target_size, embedding_dim = embedding_dim)

    elif model_type == 'deep_model':

        model = build_deep_autoencoder(input_shape = target_size, embedding_dim = embedding_dim)

    elif model_type == 'deep_model_2':

        model = build_deep_autoencoder_2(input_shape = target_size, embedding_dim = embedding_dim)

    return model
X_train = []

for images in train_generator:

    X_train = images

    break
model_name = 'deep_model_2'

model = get_autoencoder_model(model_name, target_size = target_size, embedding_dim = 2048)

model.summary()
hist = model.fit(X_train, X_train, batch_size = 64, epochs =500, initial_epoch=0)
model.save_weights('{}.h5'.format(model_name))
plt.figure(figsize=(20, 5))

row = 2

col = 8

for ix in range(col):

    plt.subplot(row, col, ix + 1)

    plt.imshow(X_train[ix])

    plt.xticks([])

    plt.yticks([])

for ix in range(col,row * col):

    plt.subplot(row, col, ix + 1)

    img = model.predict(np.expand_dims(X_train[ix - col], axis = 0))[0]

    plt.imshow(img)

    plt.xticks([])

    plt.yticks([])    

    
plt.figure(figsize = (15, 7))

for key in hist.history.keys():

    plt.plot(hist.history[key], label = key)

plt.legend()