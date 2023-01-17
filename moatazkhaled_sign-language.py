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
import keras

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv('/kaggle/input/sign-language-mnist/sign_mnist_train.csv')

test = pd.read_csv('/kaggle/input/sign-language-mnist/sign_mnist_test.csv')
train.head()
train.shape
labels = train['label'].values
value_l = np.array(labels)

np.unique(value_l)
plt.figure(figsize = (18,8))

sns.countplot(x =labels)
from sklearn.preprocessing import LabelBinarizer

label_binrizer = LabelBinarizer()

labels = label_binrizer.fit_transform(labels)

labels
train.drop('label',axis=1,inplace = True)
images = train.values

print(images.dtype, np.round(images.min(), 4), np.round(images.max(), 4), images.shape)
plt.style.use('grayscale')

fig, axs = plt.subplots(1, 5, figsize=(15, 4), sharey=True)

for i in range(5): 

        axs[i].imshow(images[i].reshape(28,28))

fig.suptitle('Grayscale images')
images = images/255
from sklearn.model_selection import train_test_split



x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size = 0.3, stratify = labels, random_state = 1)

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.3, stratify = labels, random_state = 1)







print(x_train.shape[0], 'train samples')

print(x_test.shape[0], 'test samples')

print(x_val.shape[0], 'val samples')
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_val = x_test.reshape(x_test.shape[0], 28, 28, 1)
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
model = Sequential()



model.add(Conv2D(16, kernel_size=(4,4), activation = 'relu', input_shape=(28, 28 ,1), padding='same' ))

model.add(Dropout(0.2))

model.add(MaxPooling2D(pool_size = (2, 2)))



model.add(Conv2D(32, kernel_size=(4,4), activation = 'relu', input_shape=(28, 28 ,1), padding='same' ))

model.add(Dropout(0.2))

model.add(MaxPooling2D(pool_size = (2, 2)))



model.add(Conv2D(64, kernel_size=(4,4), activation = 'relu', input_shape=(28, 28 ,1), padding='same' ))

model.add(Dropout(0.2))

model.add(MaxPooling2D(pool_size = (2, 2)))



model.add(Flatten())

model.add(Dense(500, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(24, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', 

                  metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint   



# train the model

checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=1, 

                               save_best_only=True)

hist = model.fit(x_train, y_train, batch_size=32, epochs=20,

          validation_data=(x_val, y_val), callbacks=[checkpointer], 

          verbose=2, shuffle=True)

          

          

          
model.load_weights('model.weights.best.hdf5')
test_labels = test['label'].values

test.drop('label', axis = 1, inplace = True)

test_images = test.values/255

test_images = np.array([np.reshape(i, (28, 28)) for i in test_images])

test_images = np.array([i.flatten() for i in test_images])

test_labels = label_binrizer.fit_transform(test_labels)

test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)





test_images.shape
y_pred = model.predict(test_images)

from sklearn.metrics import accuracy_score

y_pred = y_pred.round()

accuracy_score(test_labels, y_pred)