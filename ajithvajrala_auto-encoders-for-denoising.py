import numpy as np 

import pandas as pd 

from keras.layers import Input, Dense

from keras.models import Model

from keras.datasets import mnist

import matplotlib.pyplot as plt

%matplotlib inline

from skimage.io import imread, imshow, imsave

from keras.preprocessing.image import load_img, array_to_img, img_to_array

from keras.models import Sequential, Model

from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Input

from keras.optimizers import Adam, Adadelta, Adagrad

from keras import backend as K

from sklearn.model_selection import train_test_split

import glob

import warnings

warnings.filterwarnings('ignore')
train_images = glob.glob('../input/train/*.png')

train_labels = glob.glob('../input/train_cleaned/*.png')

test_images = glob.glob('../input/test/*.png')



print("Total number of images in the training set: ", len(train_images))

print("Total number of cleaned images found: ", len(train_labels))

print("Total number of samples in the test set: ", len(test_images))
samples = train_images[:3] + train_labels[:3]

f, ax = plt.subplots(2, 3, figsize=(20,10))

for i, img in enumerate(samples):

    img = imread(img)

    ax[i//3, i%3].imshow(img, cmap='gray')

    ax[i//3, i%3].axis('off')

plt.show()    
X = []

Y = []

X_test = []



for img in train_images:

    img = load_img(img, grayscale=True,target_size=(420,540))

    img = img_to_array(img).astype('float32')/255.

    X.append(img)

for img in train_labels:

    img = load_img(img, grayscale=True,target_size=(420,540))

    img = img_to_array(img).astype('float32')/255.

    Y.append(img)



for img in test_images:

    img = load_img(img, grayscale=True,target_size=(420,540))

    img = img_to_array(img).astype('float32')/255.

    X_test.append(img)





X = np.array(X)

Y = np.array(Y)

X_test = np.array(X_test)



print("Size of X : ", X.shape)

print("Size of Y : ", Y.shape)

print("Size of X_test : ", X_test.shape)
input_img = Input(shape=(420, 540, 1))  



x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)

x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)

encoded = MaxPooling2D((2, 2), padding='same')(x)





x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)

x = UpSampling2D((2, 2))(x)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)

x = UpSampling2D((2, 2))(x)

decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)



autoencoder = Model(input_img, decoded)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size = 0.2, random_state=49)

print("Total number of training samples: ", X_train.shape)

print("Total number of validation samples: ", X_val.shape)
autoencoder.summary()
autoencoder.fit(X_train, y_train,

                epochs=10,

                batch_size=8,

                shuffle=True,

                validation_data=(X_val, y_val))
X_test[0].shape
sample = np.expand_dims(X_test[1], axis=0)

predicted_label = np.squeeze(autoencoder.predict(sample))



f, ax = plt.subplots(1,2, figsize=(10,8))

ax[0].imshow(np.squeeze(sample), cmap='gray')

ax[1].imshow(np.squeeze(predicted_label.astype('int8')), cmap='gray')

plt.show()