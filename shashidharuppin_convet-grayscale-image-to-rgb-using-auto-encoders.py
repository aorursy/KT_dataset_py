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
from skimage.color import rgb2lab, lab2rgb, gray2rgb

from skimage.io import imsave

from skimage.transform import resize

import glob

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.layers import Conv2D, Flatten

from tensorflow.keras.layers import Reshape, Conv2DTranspose, Conv2D, Flatten, Reshape, MaxPooling2D, UpSampling2D

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.utils import plot_model

from tensorflow.keras import backend as K

import keras

import cv2

import numpy as np

import matplotlib.pyplot as plt

import os
#Create a data generator with some image augmentation.

input_path = '../input/train_data'

train_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

    input_path,

    batch_size=64,

    target_size=(224,224),

    color_mode="rgb",

    class_mode=None)
valid_path = '../input/valid_data'

valid_datagen = ImageDataGenerator(rescale=1./255)



valid_generator = valid_datagen.flow_from_directory(

    valid_path,

    target_size=(224,224),

    color_mode="rgb",

    class_mode=None)
train = []

files = glob.glob("../input/train_data/train/*.jpg") # your image path



for myFile in files:

    train_img = load_img(myFile,target_size=(224,224),color_mode="rgb")

    train_img= img_to_array(train_img)

    train_img = train_img / 255

    train.append(train_img)



for i in range(1,10):

    plt.subplot(330 + 0 + i)

    plt.imshow(array_to_img(train[i]))

plt.show()
#Geting the images into Lab color space and taking L, a, b channels seperately 

X = []

Y = []

for img in train:

    try:

        lab = rgb2lab(img)

        X.append(lab[:,:,0])

        Y.append(lab[:,:,1:] / 128) #a,b channel range is from -127 to 128

    

    except:

        print('Error')

        

X = np.array(X)

Y = np.array(Y)



X = X.reshape(X.shape+(1,))



print(X.shape) #L-channel

print(Y.shape) #a,b channels
plt.imshow(array_to_img(X[0]))
plt.imshow(array_to_img(train[0]))
test = []

fikes = glob.glob("../input/valid_data/valid/*.jpg")

for myfile in fikes:

    test_img = load_img(myfile,target_size=(224,224),color_mode="rgb")

    test_img= img_to_array(test_img)

    test_img = test_img / 255

    test.append(test_img)



for i in range(1,10):

    plt.subplot(330 + 0 + i)

    plt.imshow(array_to_img(test[i]))

plt.show()
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)
print(len(x_train))

print(len(y_train))

print(len(x_test))

print(len(y_test))
x_train = np.asarray(x_train)

print("The Shape of Train array is",x_train.shape)



y_train = np.asarray(y_train)

print("The Shape of Train output array is",y_train.shape)



x_test = np.asarray(x_test)

print("The Shape of Test array is",x_test.shape)



y_test = np.asarray(y_test)

print("The Shape of Test output array is",y_test.shape)
img_rows = 224

img_cols = 224

channels = 3
# Creating Encoder Model



latent_dim = 1000

input_img = Input(shape = (img_rows,img_cols,1), name='encoder_input')



conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) 

pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) 

conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) 

pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) 

conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)

pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)

pool4 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv4)

conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)

encoded = MaxPooling2D(pool_size=(2, 2), padding='same')(conv5)



shape = K.int_shape(encoded)





# generate a latent vector

x = Flatten()(encoded)

fc1 = Dense(4096, name ='FC-1')(x)

fc2 = Dense(4096, name = 'FC-2')(fc1)

latent = Dense(latent_dim, name='latent_vector')(fc2)



# instantiate encoder model

encoder = Model(input_img, latent, name='encoder')

encoder.summary()
#Creating Decoder Model



latent_inputs = Input(shape=(latent_dim,), name='decoder_input')

x = Dense(shape[1]*shape[2]*shape[3])(latent_inputs)

x = Reshape((shape[1], shape[2], shape[3]))(x)



conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(x)

up1 = UpSampling2D((2,2))(conv6)

conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up1)

up2 = UpSampling2D((2, 2))(conv7)

conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up2)

up3 = UpSampling2D((2, 2))(conv8)

conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up3)

up4 = UpSampling2D((2, 2))(conv9)

conv10 = Conv2D(32, (3, 3), activation='relu', padding='same')(up4)

up5 = UpSampling2D((2, 2))(conv10)

decoded = Conv2D(2, (3, 3), activation='tanh', padding='same')(up5)



# instantiate decoder model

decoder = Model(latent_inputs, decoded, name='decoder')

decoder.summary()
import keras

from keras import optimizers

#Fitting the model

autoencoder = Model(input_img, decoder(encoder(input_img)))

#adam = optimizers.Adam(learning_rate=0.001,amsgrad=False)

autoencoder.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])
autoencoder.fit(x_train,y_train,verbose = 1, epochs = 30,batch_size = 128, validation_data=(x_test,y_test))
x_decoded = autoencoder.predict(x_test)
from keras.preprocessing.image import img_to_array

from keras.preprocessing.image import array_to_img



plt.imshow(array_to_img(x_test[0]))
ab = x_decoded[0] * 128

l = x_test[0]

l = np.reshape(l,(224,224))

reconstructed_img = np.zeros((224, 224, 3))

reconstructed_img[:, :, 0]= l

reconstructed_img[:, :, 1:] = ab

plt.imshow(lab2rgb(reconstructed_img))

#imsave('reconstructed_img'+ str(ids)+".png" ,lab2rgb(reconstructed_img))
ab = y_test[0] * 128

l = x_test[0]

l = np.reshape(l,(224,224))

reconstructed_img = np.zeros((224, 224, 3))

reconstructed_img[:, :, 0]= l

reconstructed_img[:, :, 1:] = ab

plt.imshow(lab2rgb(reconstructed_img))    
for i in range(1,4):

    fig = plt.figure(figsize=(10,10))

    

    ab = x_decoded[i] * 128

    l = x_test[i]

    l = np.reshape(l,(224,224))

    reconstructed_img = np.zeros((224, 224, 3))

    reconstructed_img[:, :, 0]= l

    reconstructed_img[:, :, 1:] = ab

    

    bc = y_test[i] * 128

    original_img = np.zeros((224, 224, 3))

    original_img[:, :, 0]= l

    original_img[:, :, 1:] = bc

    

    ax1 = fig.add_subplot(3,3,1)

    ax1.set_title("Gray Scale")

    ax1.imshow(array_to_img(x_test[i]))

    

    ax2 = fig.add_subplot(3,3,2)

    ax2.set_title("Ground_Truth")

    ax2.imshow(lab2rgb(original_img))

    

    ax3 = fig.add_subplot(3,3,3)

    ax3.set_title("Converted RGB")

    ax3.imshow(lab2rgb(reconstructed_img))
for i in range(1,11):

    ab = x_decoded[i] * 128

    l = x_test[i]

    l = np.reshape(l,(224,224))

    reconstructed_img = np.zeros((224, 224, 3))

    reconstructed_img[:, :, 0]= l

    reconstructed_img[:, :, 1:] = ab

    imsave('reconstructed_img'+ str(i)+".png" ,lab2rgb(reconstructed_img))