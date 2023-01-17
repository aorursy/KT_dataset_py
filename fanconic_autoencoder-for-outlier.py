import os



%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import os

from glob import glob

import seaborn as sns

from PIL import Image

np.random.seed(11) # It's my lucky number

from sklearn.preprocessing import StandardScaler 

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV

from sklearn.metrics import accuracy_score

import itertools



import keras

from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding

from keras.models import Sequential, Model

from keras.layers import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D, UpSampling2D

from keras import backend as K

from keras.layers.normalization import BatchNormalization

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.optimizers import Adam, RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from keras.wrappers.scikit_learn import KerasClassifier

from keras.applications.resnet50 import ResNet50

from keras import backend as K 



import tensorflow as tf
# Load the extension and start TensorBoard



%load_ext tensorboard.notebook

%tensorboard --logdir logs
os.listdir('../input/random-images')
folder_benign = '../input/skin-cancer-malignant-vs-benign/data/data/benign'

folder_malignant = '../input/skin-cancer-malignant-vs-benign/data/data/malignant'

folder_outliers = '../input/random-images/'

read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))



# Load in pictures

ims_benign = [read(os.path.join(folder_benign, filename)) for filename in os.listdir(folder_benign)]

X_benign = np.array(ims_benign, dtype='uint8')

ims_malignant = [read(os.path.join(folder_malignant, filename)) for filename in os.listdir(folder_malignant)]

X_malignant = np.array(ims_malignant, dtype='uint8')

ims_outliers = [read(os.path.join(folder_outliers, filename)) for filename in os.listdir(folder_outliers)]

X_outliers = np.array(ims_outliers, dtype='uint8')



# Create labels

y_benign = np.zeros(X_benign.shape[0])

y_malignant = np.zeros(X_malignant.shape[0])

y_outliers = np.ones(X_outliers.shape[0])



# Merge data and shuffle it

X = np.concatenate((X_benign, X_malignant), axis = 0)

y = np.concatenate((y_benign, y_malignant), axis = 0)

s = np.arange(X.shape[0])

np.random.shuffle(s)

X = X[s]

y = y[s]
# With data augmentation to prevent overfitting 

X_scaled = X/255.

X_outliers = X_outliers/255.
X_train, X_test, y_train, y_test= train_test_split(X_scaled, 

                                                 y,

                                                 test_size=0.20,

                                                 random_state=42)
input_img = Input(shape=(224, 224, 3))  # adapt this if using `channels_first` image data format



x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)

x = MaxPooling2D((4, 4), padding='same')(x)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)

x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)

encoded = MaxPooling2D((2, 2), padding='same')(x)





x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)

x = UpSampling2D((2, 2))(x)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)

x = UpSampling2D((2, 2))(x)

x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)

x = UpSampling2D((4, 4))(x)

decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)





autoencoder = Model(input_img, decoded)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.summary()
tensorboard_callback = tf.keras.callbacks.TensorBoard("logs")



autoencoder.fit(X_train, X_train,

                epochs=10,

                batch_size=128,

                shuffle=True,

                validation_data=(X_test, X_test),

                callbacks=[tensorboard_callback])
decoded_imgs = autoencoder.predict(X_test)



n = 10

plt.figure(figsize=(20, 4))

for i in range(1,n+1):

    # display original

    ax = plt.subplot(2, n, i)

    plt.imshow(X_test[i].reshape(224, 224,3))

    plt.plot()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)



    # display reconstruction

    ax = plt.subplot(2, n, i + n)

    plt.imshow(decoded_imgs[i].reshape(224, 224, 3))

    plt.plot()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

plt.show()



decoded_outliers = autoencoder.predict(X_outliers)



n = 5

plt.figure(figsize=(20, 4))

for i in range(n):

    # display original

    ax = plt.subplot(2, n, i+1)

    plt.imshow(X_outliers[i].reshape(224, 224,3))

    plt.plot()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)



    # display reconstruction

    ax = plt.subplot(2, n, i + n+1)

    plt.imshow(decoded_outliers[i].reshape(224, 224, 3))

    plt.plot()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

plt.show()
decoded_train = autoencoder.predict(X_train)



# Check average MSE for every mole in train set

mses_train = []

for i in range(decoded_imgs.shape[0]):

    mse = np.mean((X_train[i] - decoded_train[i]) ** 2 )

    mses_train.append(mse)

print('Average MSE for mole train: ', np.mean(np.asarray(mses_train)))



# Check average MSE for every mole in test set

mses_moles = []

for i in range(decoded_imgs.shape[0]):

    mse = np.mean((X_test[i] - decoded_imgs[i]) ** 2 )

    mses_moles.append(mse)

print('Average MSE for mole test: ', np.mean(np.asarray(mses_moles)))



# Check average MSE for every outlier

mses_outliers = []

for i in range(decoded_outliers.shape[0]):

    mse = np.mean((X_outliers[i] - decoded_outliers[i]) ** 2 )

    mses_outliers.append(mse)

print('Average MSE for outliers: ', np.mean(np.asarray(mses_outliers)))
max(mses_train), max(mses_moles), min(mses_outliers)
threshold_fixed = 0.04

fig, ax = plt.subplots()



# Plot threshold line

for i in range(len(mses_train)):

    ax.plot(i, mses_train[i], marker='o', ms=3.5, linestyle='')

for i in range(len(mses_moles)):

    ax.plot(i, mses_moles[i], marker='o', ms=3.5, linestyle='')

for i in range(len(mses_outliers)):

    ax.plot(i, mses_outliers[i], marker='o', ms=3.5, linestyle='')

ax.hlines(threshold_fixed, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')

ax.legend()

plt.title("Reconstruction error for different classes")

plt.ylabel("Reconstruction error")

plt.xlabel("Data point index")

plt.show();
# save model

# serialize model to JSON

model_json = autoencoder.to_json()



with open("autoencoder.json", "w") as json_file:

    json_file.write(model_json)

    

# serialize weights to HDF5

autoencoder.save_weights("autoencoder.h5")

print("Saved model to disk")