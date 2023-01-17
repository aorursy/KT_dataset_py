# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from keras.utils import to_categorical

import matplotlib.pyplot as plt



import keras

from keras.models import Sequential, load_model

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.layers.normalization import BatchNormalization

from keras.layers.advanced_activations import LeakyReLU

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
X = np.load("../input/sign-language-digits-dataset/Sign-language-digits-dataset/X.npy")

y = np.load("../input/sign-language-digits-dataset/Sign-language-digits-dataset/Y.npy")
X = np.reshape(X,(-1,64,64,1))

# Split 2062 samples into 1717 and 345

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.167, random_state = 1)

# Split 1649 samples into 1373 and 344

X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size = 0.2, random_state = 1)
print('Training data shape : ', X_train.shape, y_train.shape)

print('Dev data shape : ', X_dev.shape, y_dev.shape)

print('Test data shape : ', X_test.shape, y_test.shape)



# print(y_dev.sum(axis=0))

# print(y_test.sum(axis=0))
batch_size = 64

epochs = 100

num_classes = 10



dr = Sequential()

dr.add(Conv2D(6, kernel_size=(3,3),activation='linear',input_shape=(64,64,1),padding='same'))

dr.add(BatchNormalization(axis=-1))

dr.add(LeakyReLU(alpha=0.1))

dr.add(MaxPooling2D((2,2),padding='same'))

dr.add(Dropout(0.2))

dr.add(Conv2D(16, (3,3), activation='linear',padding='same'))

dr.add(BatchNormalization(axis=-1))

dr.add(LeakyReLU(alpha=0.1))

dr.add(MaxPooling2D(pool_size=(2,2),padding='same'))

dr.add(Dropout(0.2))

dr.add(Conv2D(64, (3,3), activation='linear',padding='same'))

dr.add(BatchNormalization(axis=-1))

dr.add(LeakyReLU(alpha=0.1))                  

dr.add(MaxPooling2D(pool_size=(2,2),padding='same'))

dr.add(Dropout(0.3))

dr.add(Flatten())

dr.add(Dense(60, activation='linear'))

dr.add(BatchNormalization(axis=-1))

dr.add(LeakyReLU(alpha=0.1))         

dr.add(Dropout(0.2))         

dr.add(Dense(25, activation='linear'))

dr.add(BatchNormalization(axis=-1))

dr.add(LeakyReLU(alpha=0.1))         

dr.add(Dropout(0.2)) 

dr.add(Dense(num_classes, activation='softmax'))



dr.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])



dr.summary()



training = dr.fit(X_train, y_train, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(X_dev, y_dev))



dr.save("Conv2D_Signs.h5py")



test_eval = dr.evaluate(X_dev, y_dev, verbose=0)

print(test_eval)



accuracy = training.history['accuracy']

val_accuracy = training.history['val_accuracy']

loss = training.history['loss']

val_loss = training.history['val_loss']

epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'bo', label='Training accuracy')

plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()



from IPython.display import FileLink

FileLink(r"Conv2D_Signs.h5py")
print("Training dataset evaluation")

test_eval = dr.evaluate(X_train, y_train, verbose=0)

print(test_eval)



print("Dev dataset evaluation")

test_eval = dr.evaluate(X_dev, y_dev, verbose=0)

print(test_eval)



print("Test dataset evaluation")

test_eval = dr.evaluate(X_test, y_test, verbose=0)

print(test_eval)