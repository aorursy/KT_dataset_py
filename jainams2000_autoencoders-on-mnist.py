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
from scipy.io import loadmat

mnist = loadmat("/kaggle/input/mnist-original/mnist-original.mat")
mnist_data = mnist['data'].T
mnist_data.shape
import os,shutil,glob

from tqdm.notebook import tqdm

from skimage.transform import resize,rescale

import matplotlib.pyplot as plt

import matplotlib.image as mpimg 



import numpy as np
resized_mnist = []

for img in tqdm(mnist_data):

    resized_mnist.append(np.reshape(img,(28,28,1)))
np.shape(resized_mnist)
from sklearn.model_selection import train_test_split



X_train, X_test = train_test_split(

    resized_mnist,

    test_size=0.2,

    shuffle=True,

    random_state=42,

)

# X_train, X_val = train_test_split(

#     X_train, 

#     test_size=0.2,

#     shuffle=True,

#     random_state=42,)



X_train = np.array(X_train)

X_test = np.array(X_test)

# X_val = np.array(X_val)



print(X_train.shape,X_test.shape)
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D

from tensorflow.keras.models import Model
autoencoder = None

input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format



x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)

x = MaxPooling2D((2, 2), padding='same')(x)

# x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)

# x = MaxPooling2D((2, 2), padding='same')(x)

# x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)

# x = MaxPooling2D((2, 2), padding='same')(x)

# x = Dense(32, activation='relu')(x)

# x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)

# x = UpSampling2D((2, 2))(x)

# x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)

# x = UpSampling2D((2, 2))(x)

# x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)

x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)

x = UpSampling2D((2, 2))(x)

# x = UpSampling2D((2, 2))(x)

decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# -------------------------------------------------------------------------------------------



autoencoder = Model(input_img, decoded)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy',

              metrics=['accuracy'])



autoencoder.summary()
history = autoencoder.fit(X_train, X_train,

                epochs=50,

                batch_size=128,

                shuffle=True,

                validation_data=(X_test, X_test))
n = 10

plt.figure(figsize=(20, 4))

for i in range(1,n):

    # display original

    ax = plt.subplot(1, n, i)

    plt.imshow(X_train[i])

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

plt.show()