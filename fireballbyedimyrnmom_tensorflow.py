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
from __future__ import absolute_import, division, print_function, unicode_literals





import tensorflow as tf

from tensorflow import keras # tf.keras is a high-level API to build and train models in TensorFlow



import numpy as np

import matplotlib.pyplot as plt



print(tf.__version__) #verify tensorFlow version

from keras.utils import to_categorical

import numpy as np

from sklearn.model_selection import train_test_split
# Get the Fashion MNIST data 

import pandas as pd

fashion_mnist_test = pd.read_csv("../input/fashionmnist/fashion-mnist_test.csv")

fashion_mnist_train = pd.read_csv("../input/fashionmnist/fashion-mnist_train.csv")
print(fashion_mnist_test.shape, fashion_mnist_train.shape)
#store the label names 

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',

               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
img_rows, img_cols = 28, 28

input_shape = (img_rows, img_cols, 1)



X = np.array(fashion_mnist_train.iloc[:, 1:])

y = to_categorical(np.array(fashion_mnist_train.iloc[:, 0]))
#split the validation data to optimize classifier during training

##source:https://www.kaggle.com/bugraokcu/cnn-with-keras

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=13)
#repeat for the test dataset

X_test = np.array(fashion_mnist_test.iloc[:, 1:])

y_test = to_categorical(np.array(fashion_mnist_test.iloc[:, 0]))
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)

X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)



X_train = X_train.astype('float32')

X_test = X_test.astype('float32')

X_val = X_val.astype('float32')

X_train /= 255

X_test /= 255

X_val /= 255
import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.layers.normalization import BatchNormalization



batch_size = 256

num_classes = 10

epochs = 50



#input image dimensions

img_rows, img_cols = 28, 28



model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 kernel_initializer='he_normal',

                 input_shape=input_shape))

#hidden layers

model.add(MaxPooling2D((2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(Dropout(0.4))

##Flatten layer

model.add(Flatten())

model.add(Dense(128, activation='relu'))



#output layer

model.add(Dropout(0.3))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adam(),

              metrics=['accuracy'])
model.summary()
#train

##this takes some time to run...

history = model.fit(X_train, y_train,

          batch_size=batch_size,

          epochs=epochs,

          verbose=1,

          validation_data=(X_val, y_val))

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])

print('Test accuracy:', score[1])
#predictions

predicted_classes = model.predict_classes(X_test)



#get indices from the testset for plotting

y_true = fashion_mnist_test.iloc[:, 0]

correct = np.nonzero(predicted_classes==y_true)[0]

incorrect = np.nonzero(predicted_classes!=y_true)[0]

#classification report

from sklearn.metrics import classification_report

target_names = ["Class {}".format(i) for i in range(num_classes)]

print(classification_report(y_true, predicted_classes, target_names=target_names))
#plot a subset of correctly predicted classes

for i, correct in enumerate(correct[:9]):

    plt.subplot(3,3,i+1)

    plt.imshow(X_test[correct].reshape(28,28), cmap='gray', interpolation='none')

    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_true[correct]))

    plt.tight_layout()
#a subset of incorrectly predicted classes

for i, incorrect in enumerate(incorrect[0:9]):

    plt.subplot(3,3,i+1)

    plt.imshow(X_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')

    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_true[incorrect]))

    plt.tight_layout()