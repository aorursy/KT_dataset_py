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
from tensorflow.keras.datasets import mnist

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape

import matplotlib.pyplot as plt

(X_train, _), (X_test, _) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28,28,1).astype('float32')/255

X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')/255
autoencoder = Sequential()



autoencoder.add(Conv2D(16, kernel_size=3, padding = 'same', input_shape=(28,28,1), activation='relu'))

autoencoder.add(MaxPooling2D(pool_size=2, padding = 'same'))

autoencoder.add(Conv2D(8, kernel_size =3, activation='relu', padding ='same'))

autoencoder.add(MaxPooling2D(pool_size=2, padding='same'))

autoencoder.add(Conv2D(8, kernel_size=3, strides=2, padding='same', activation='relu'))



autoencoder.add(Conv2D(8, kernel_size=3, padding='same', activation='relu'))

autoencoder.add(UpSampling2D())

autoencoder.add(Conv2D(8, kernel_size=3, padding='same', activation='relu'))

autoencoder.add(UpSampling2D())

autoencoder.add(Conv2D(16, kernel_size=3, activation='relu'))

autoencoder.add(UpSampling2D())

autoencoder.add(Conv2D(1, kernel_size=3, padding='same', activation='sigmoid'))



autoencoder.summary()
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(X_train, X_train, epochs=50, batch_size=128, validation_data=(X_test, X_test))



random_test = np.random.randint(X_test.shape[0], size=5)



ae_imgs = autoencoder.predict(X_test)



plt.figure(figsize=(7,2))





for i, image_idx in enumerate(random_test):

    ax=plt.subplot(2,7, i+1)

    plt.imshow(X_test[image_idx].reshape(28,28))

    ax.axis('off')

    ax=plt.subplot(2, 7,7+i+1)

    

    plt.imshow(ae_imgs[image_idx].reshape(28,28))

    ax.axis('off')

    

plt.show()