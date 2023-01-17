from __future__ import print_function

import keras

import tensorflow as tf

import pandas as pd

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from sklearn.model_selection import train_test_split

from keras import backend as K

import matplotlib.pyplot as plt

import numpy as np
tf.logging.set_verbosity(tf.logging.ERROR)
batch_size = 32

num_classes = 10

epochs = 36

img_rows, img_cols = 36, 36

imput_shape = (28, 28, 1)
training = pd.read_csv("../input/train.csv")

test = training["label"]

training.drop(["label"], inplace = True, axis = 1)
x_train, x_test, y_train, y_test = train_test_split(training.values, test.values, test_size=0.2 , random_state=42)
x_train = x_train.reshape(x_train.shape[0], 28, 28 , 1).astype('float32')

x_test = x_test.reshape(x_test.shape[0], 28, 28 , 1).astype('float32')
first_image = x_train[1]

first_image = np.array(first_image, dtype='float')

pixels = first_image.reshape((28, 28))

plt.imshow(pixels, cmap='gray')

plt.show()
x_train /= 255

x_test /= 255
print('x_train shape:', x_train.shape)

print(x_train.shape[0], 'train samples')

print(x_test.shape[0], 'test samples')
y_train = keras.utils.to_categorical(y_train, num_classes)

y_test = keras.utils.to_categorical(y_test, num_classes)
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(256, activation='relu'))

model.add(Dropout(0.21))

model.add(Dense(num_classes, activation='relu'))
model.add(Dense(512, activation='relu'))

model.add(Dropout(0.22))

model.add(Dense(num_classes, activation='softmax'))
model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(num_classes, activation='relu'))
model.add(Dense(128, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adam(),

              metrics=['accuracy'])
history = model.fit(x_train, 

                    y_train,

                    batch_size=batch_size,

                    epochs=epochs,

                    verbose=1,

                    validation_data=(x_test, y_test))
history.history
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model acc')

plt.ylabel('acc')

plt.xlabel('epoch')

plt.legend(['acc', 'val_acc'], loc='upper left')
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model acc')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['loss', 'val_loss'], loc='upper left')
score = model.evaluate(x_test, y_test, verbose=0)
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #for plotting

from collections import Counter

from sklearn.metrics import confusion_matrix

import itertools

import seaborn as sns

from subprocess import check_output
# Predict the values from the validation dataset

Y_pred = model.predict(x_test)

Y_pred_classes = np.argmax(Y_pred, axis = 1) 

Y_true = np.argmax(y_test, axis = 1) 



confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

plot_confusion_matrix(confusion_mtx, classes = range(10))