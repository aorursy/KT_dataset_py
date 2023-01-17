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
import glob

import numpy as np

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
train = []

files = glob.glob("../input/train/*.jpg") # your image path

for myFile in files:

    train_img = load_img(myFile,target_size=(128,128),color_mode="rgb")

    train_img= img_to_array(train_img)

    train_img = train_img / 255

    train.append(train_img)



train_tar = []

for myFile in files:

    train_img = load_img(myFile,target_size=(128,128),color_mode="grayscale")

    train_img= img_to_array(train_img)

    train_img = train_img / 255

    train_tar.append(train_img)

    

test = []

fikes = glob.glob("../input/valid/*.jpg")

for myfile in fikes:

    test_img = load_img(myfile,target_size=(128,128),color_mode="rgb")

    test_img = img_to_array(test_img)

    test_img = test_img / 255

    test.append(test_img)

    

test_tar = []

for myfile in fikes:

    test_img = load_img(myfile,target_size=(128,128),color_mode="grayscale")

    test_img = img_to_array(test_img)

    test_img = test_img / 255

    test_tar.append(test_img)



print("Length of Train is:",len(train))

print("Length of Test is:",len(test))
import matplotlib.pyplot as plt



plt.imshow(array_to_img(test[67]))

plt.show()
train = np.asarray(train)

print("The Shape of Train array is",train.shape)



train_tar = np.asarray(train_tar)

print("The Shape of Train grayscale array is",train_tar.shape)



test = np.asarray(test)

print("The Shape of Test array is",test.shape)



test_tar = np.asarray(test_tar)

print("The Shape of Test grayscale array is",test_tar.shape)
import keras

from keras.models import Model

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Flatten, Reshape,BatchNormalization

from keras import optimizers

from tensorflow.keras import backend as K





# creating autoencoder model

input_img = Input(shape = (128,128,3), name='encoder_input')

latent_dim = 512



#encoder



conv1 = Conv2D(128, (3, 3), activation='relu', padding='same')(input_img) 

pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) 

conv2 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool1) 

pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) 

conv3 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool2)

#encoded = MaxPooling2D(pool_size=(2, 2), padding='same')(conv3)



shape = K.int_shape(conv3)





# generate a latent vector

x = Flatten()(conv3)

latent = Dense(latent_dim, name='latent_vector')(x)



# instantiate encoder model

encoder = Model(input_img, latent, name='encoder')

encoder.summary()





#decoder

latent_inputs = Input(shape=(latent_dim,), name='decoder_input')

x = Dense(shape[1]*shape[2]*shape[3])(latent_inputs)

x = Reshape((shape[1], shape[2], shape[3]))(x)



#up1 = UpSampling2D((2,2))(encoded)

conv4 = Conv2DTranspose(512,(3, 3), activation='relu', padding='same')(x) 

up2 = UpSampling2D((2,2))(conv4) 

conv5 = Conv2DTranspose(256,(3, 3), activation='relu', padding='same')(up2) 

up3 = UpSampling2D((2,2))(conv5) 

conv6 = Conv2DTranspose(64,(3, 3), activation='relu', padding='same')(up3) 

decoded = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(conv6)



# instantiate decoder model

decoder = Model(latent_inputs, decoded, name='decoder')

decoder.summary()
# autoencoder = encoder + decoder

# instantiate autoencoder model

autoencoder = Model(input_img, decoder(encoder(input_img)), name='autoencoder')

autoencoder.summary()
from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.callbacks import ModelCheckpoint

import numpy as np





# reduce learning rate by sqrt(0.1) if the loss does not improve in 5 epochs

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),

                               cooldown=0,

                               patience=5,

                               verbose=1,

                               min_lr=0.5e-6)



sgd = optimizers.RMSprop(learning_rate=0.001)

# Mean Square Error (MSE) loss function, Adam optimizer

autoencoder.compile(loss='mse', optimizer= sgd)



# called every epoch

callbacks = [lr_reducer]



# train the autoencoder

autoencoder.fit(train,

                train_tar,

                validation_split=0.20,

                epochs=30,

                batch_size=32

                )
# predict the autoencoder output from test data

x_decoded = autoencoder.predict(test)
for i in range(0,6):

    fig = plt.figure()

#   fig.suptitle("RGB      GroundTruth    Converted",size=16)

    ax1 = fig.add_subplot(3,3,1)

    ax1.set_title("RGB")

    ax1.imshow(array_to_img(test[i]))

    

    ax2 = fig.add_subplot(3,3,2)

    ax2.set_title("Ground_Truth")

    ax2.imshow(array_to_img(test_tar[i]))

    

    ax3 = fig.add_subplot(3,3,3)

    ax3.set_title("Converted Gray Scale")

    ax3.imshow(array_to_img(x_decoded[i]))