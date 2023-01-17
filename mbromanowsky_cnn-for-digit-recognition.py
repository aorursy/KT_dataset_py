# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import confusion_matrix
import itertools

sns.set(style='white', context='notebook', palette='deep')

np.random.seed(5)
# Data prep code carried over from the Deep Learning lessons

img_rows, img_cols = 28, 28
num_classes = 10

def data_prep(raw):
    out_y = keras.utils.to_categorical(raw.label, num_classes)

    num_images = raw.shape[0]
    x_as_array = raw.values[:,1:]
    x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1)
    out_x = x_shaped_array / 255.0
    return out_x, out_y

train_file = "../input/train.csv"
raw_data = pd.read_csv(train_file)

x, y = data_prep(raw_data)
model = Sequential()
model.add(Conv2D(20, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))
model.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])
model.fit(x, y,
          batch_size=128,
          epochs=2,
          validation_split = 0.2)
model2 = Sequential()
model2.add(Conv2D(20, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))
model2.add(Conv2D(20, kernel_size=(5, 5), activation='relu'))
model2.add(Flatten())
model2.add(Dense(128, activation='relu'))
model2.add(Dense(num_classes, activation='softmax'))

model2.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])
model2.fit(x, y,
          batch_size=128,
          epochs=2,
          validation_split = 0.2)
model3 = Sequential()
model3.add(Conv2D(50, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))
model3.add(Conv2D(50, kernel_size=(5, 5), activation='relu'))
model3.add(Flatten())
model3.add(Dense(128, activation='relu'))
model3.add(Dense(num_classes, activation='softmax'))

model3.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])
model3.fit(x, y,
          batch_size=128,
          epochs=2,
          validation_split = 0.2)
model4 = Sequential()
model4.add(Conv2D(100, kernel_size=(10, 10),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))
model4.add(Conv2D(100, kernel_size=(10, 10), activation='relu'))
model4.add(Flatten())
model4.add(Dense(128, activation='relu'))
model4.add(Dense(num_classes, activation='softmax'))

model4.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])
model4.fit(x, y,
          batch_size=128,
          epochs=2,
          validation_split = 0.2)
from tensorflow.python.keras.layers import Dropout

model5 = Sequential()
model5.add(Conv2D(100, kernel_size=(10, 10),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))
model5.add(Dropout(0.5))
model5.add(Conv2D(100, kernel_size=(10, 10), activation='relu'))
model5.add(Dropout(0.5))
model5.add(Flatten())
model5.add(Dense(128, activation='relu'))
model5.add(Dense(num_classes, activation='softmax'))

model5.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])
model5.fit(x, y,
          batch_size=128,
          epochs=2,
          validation_split = 0.2)
x_test = pd.read_csv('../input/test.csv')
x_test = x_test / 255.0
x_test = x_test.values.reshape(-1,28,28,1)

g = plt.imshow(x[0][:,:,0])
g = plt.imshow(x[0][:,:,0])
%matplotlib inline
g = plt.imshow(x[0][:,:,0])
for i in range(10):
    print(f"Training observation {i}, which has label {np.argmax(y[i])}")
    plt.imshow(x[i][:,:,0])
    plt.show()
# predict based on the whole training set
y_pred = model2.predict(x)

y_pred[:10]
y_pred_classes = np.argmax(y_pred, axis=1)
y_pred_classes[:10]
y_true_classes = np.argmax(y, axis=1)
conf_mtx = confusion_matrix(y_true_classes, y_pred_classes)
conf_mtx
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
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
plot_confusion_matrix(conf_mtx, classes=range(10))
# apply model2 to test data
y_test_pred = model2.predict(x_test)
# pick highest prob class
y_test_pred_class = np.argmax(y_test_pred, axis = 1)
y_out_series = pd.Series(y_test_pred_class, name='Label')
submission = pd.concat([pd.Series(range(1,28001), name='ImageId'), y_out_series], axis=1)
submission.to_csv('model2_sub.csv', index=False)
# apply model5 to test data, and see if it's any different / better

y_test_pred = model5.predict(x_test)
# pick highest prob class
y_test_pred_class = np.argmax(y_test_pred, axis = 1)
y_out_series = pd.Series(y_test_pred_class, name='Label')
submission = pd.concat([pd.Series(range(1,28001), name='ImageId'), y_out_series], axis=1)
submission.to_csv('model5_sub.csv', index=False)