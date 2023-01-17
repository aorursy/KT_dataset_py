# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import *

import matplotlib.pyplot as plt

import seaborn as sns

import cv2 as cv2

from sklearn.model_selection import train_test_split



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv('../input/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv')

train_df.shape
train_df.head()
test_df = pd.read_csv('../input/sign-language-mnist/sign_mnist_test/sign_mnist_test.csv')

test_df.shape
test_df.head()
train_df.isnull().sum()
test_df.isnull().sum()
train_df.dtypes
train_df['label'].values
labels = train_df['label'].values
unique_set = np.unique(np.array(labels))
plt.figure(figsize=(10,10))

sns.set(style="darkgrid")

sns.countplot(y=labels, data=train_df, palette='Set2')
train_df.drop(['label'],axis=1,inplace=True)
img = cv2.imread('../input/sign-language-mnist/amer_sign2.png')

plt.imshow(img)
img = cv2.imread('../input/sign-language-mnist/american_sign_language.PNG')

plt.imshow(img)
images = train_df.values

images = np.array([np.reshape (i, (28,28)) for i in images])

images = np.array([i.flatten() for i in images])
plt.imshow(images[0].reshape(28, 28))
from sklearn.preprocessing import LabelBinarizer



lb = LabelBinarizer()

labels = lb.fit_transform(labels)
labels[:5]
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=140)



print('Training Data shape : ',x_train.shape,  y_train.shape)

print('Testing Data shape : ',x_test.shape,  y_test.shape)
batch_size=256

EPOCHS = 50
x_train, x_test = x_train.astype(np.float32), x_test.astype(np.float32)

# Flatten images to 1-D vector of 784 features (28*28).

x_train, x_test = x_train.reshape([-1, 784]), x_test.reshape([-1, 784])

# Normalize images value from [0, 255] to [0, 1].



x_train = x_train/255.

x_test = x_test/255.
plt.imshow(x_train[0].reshape(28,28)) #Since image size is 784 so (28,28)

plt.axis('off')
input_img = tf.keras.layers.Input(shape=(784,), name = "input")



# this is the encoded representation of the input

encoded = Dense(1024, activation='relu', name="emb_0")(input_img)

encoded = Dense(512, activation='relu', name="emb_1")(encoded)

encoded = Dense(256, activation='relu', name="emb_2")(encoded)

encoded = Dense(128, activation='relu', name="emb_3")(encoded)

encoded = Dense(64, activation='relu', name="emb_4")(encoded)

encoded = Dense(16, activation='relu', name="emb_5")(encoded)

latent_vector = Dense(2, activation='relu', name="latent_vector")(encoded)
# this is the loss reconstruction of the input

decoded = Dense(16, activation='relu', name="dec_1")(latent_vector)

decoded = Dense(64, activation='relu', name="dec_3")(decoded)

decoded = Dense(128, activation='relu', name="dec_4")(decoded)

decoded = Dense(256, activation='relu', name="dec_5")(decoded)

decoded = Dense(512, activation='relu', name="dec_6")(decoded)

decoded = Dense(1024, activation='relu', name="dec_7")(decoded)



output_layer = Dense(784, activation = 'sigmoid', name="output")(decoded)
autoencoder = tf.keras.models.Model(input_img, output_layer)
autoencoder.summary()
encoder = tf.keras.models.Model(input_img, latent_vector)

encoder.summary()
autoencoder.compile(optimizer='adam', loss='mse')

auto_history = autoencoder.fit(x_train, x_train, epochs=EPOCHS, batch_size=batch_size,validation_data=(x_test, x_test))
decoded_imgs = autoencoder.predict(x_test)
n = 10 

plt.figure(figsize=(20, 4))

for i in range(n):

    # display original

    ax = plt.subplot(2, n, i + 1)

    plt.imshow(x_test[i].reshape(28, 28))

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

    

    # display reconstruction

    ax = plt.subplot(2, n, i + 1 + n)

    plt.imshow(decoded_imgs[i].reshape(28, 28))

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)



plt.show()