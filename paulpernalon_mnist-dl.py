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
import numpy as np

from keras.utils import np_utils

from keras.layers import Input, Dense, Dropout, Convolution2D, MaxPooling2D, Flatten, Activation

from keras.models import Model, Sequential, load_model, save_model

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from keras import metrics

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
# nb_filters = 32

# kernel_size = (3,3)

# pool_size = (3,3)

# input_shape = (28, 28, 1)



def mnist_model(nb_classes, nb_filters, pool_size, kernel_size, input_shape):

    model = Sequential()

    model.add(Convolution2D(nb_filters, kernel_size, padding='valid', input_shape=input_shape))

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Convolution2D(nb_filters, kernel_size, padding='valid'))

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(64, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(nb_classes))

    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',

                    optimizer='adadelta',

                    metrics=['accuracy'])

    return model
def plot_learning_curves(history):

    #print history.history.keys()

    plt.plot(history.history['acc'])

    plt.plot(history.history['val_acc'])

    plt.title('model accuracy')

    plt.ylabel('accuracy')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.show()

    # summarize history for loss

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('model loss')

    plt.ylabel('loss')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.show()
from sklearn.model_selection import train_test_split



train_mnist_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test_mnist_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')



nb_classes = 10



X = np.array(train_mnist_data.iloc[:, 1:])/255

y = np.array(train_mnist_data.iloc[:, 0])



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

y_test = np_utils.to_categorical(y_test, nb_classes)

y_train = np_utils.to_categorical(y_train, nb_classes)



X_validation = np.array(test_mnist_data)/255

X_validation = X_validation.reshape(X_validation.shape[0], 28, 28, 1)



print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)

print(X_validation.shape)
nb_classes = 10

nb_filters = 32

kernel_size = (3,3)

pool_size = (2,2)

input_shape = (28, 28, 1)



mnist_clf = mnist_model(nb_classes, nb_filters, pool_size, kernel_size, input_shape)

mnist_clf.summary()
batch_size = 128

nb_epoch = 500



mnist_clf.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch)
score = mnist_clf.evaluate(X_test, y_test, verbose=0)

print('Test score:', score[0])

print('Test accuracy:', score[1])
y_validation = mnist_clf.predict_classes(X_validation)

sample_submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

sample_submission["Label"] = y_validation



output = pd.DataFrame({'ImageId': sample_submission['ImageId'], 'Label': y_validation})

output.to_csv('validation.csv', index = False)