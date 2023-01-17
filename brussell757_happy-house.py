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
import h5py

def load_dataset():
    train_data = h5py.File('../input/train_happy.h5', 'r')
    x_train = np.array(train_data['train_set_x'][:450])
    y_train = np.array(train_data['train_set_y'][:450])
    
    x_val = np.array(train_data['train_set_x'][450:])
    y_val = np.array(train_data['train_set_y'][450:])
    
    test_data = h5py.File('../input/test_happy.h5', 'r')
    x_test = np.array(test_data['test_set_x'][:])
    y_test = np.array(test_data['test_set_y'][:])
    
    y_train = y_train.reshape((y_train.shape[0], 1))
    y_val = y_val.reshape((y_val.shape[0], 1))
    y_test = y_test.reshape((y_test.shape[0], 1))
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
(x_train, y_train), (x_val, y_val), (x_test, y_test) = load_dataset()

print('Training Set X Shape: {}, Training Set Y Shape: {}'.format(x_train.shape, y_train.shape))
print('Validation Set X Shape: {}, Validation Set Y Shape: {}'.format(x_val.shape, y_val.shape))
print('Testing Set X Shape: {}, Testing Set Y Shape: {}'.format(x_test.shape, y_test.shape))
import matplotlib.pyplot as plt

plt.figure(figsize = (25, 15))
for img_index in range(10):
    plt.subplot(2, 5, img_index + 1)
    plt.imshow(x_train[img_index])
    plt.xlabel('y = {}'.format(y_train[img_index]))
x_train = x_train / 255.
x_val = x_val / 255.
x_test = x_test / 255.
from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def build_model(conv_units_1 = 16, conv_units_2 = 32, units = 128, dropout = 0.2, optimizer = 'rmsprop'):
    model = Sequential()
    model.add(Conv2D(conv_units_1, 3, activation = 'relu', input_shape = (64, 64, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2))
    model.add(Conv2D(conv_units_2, 3, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2))
    model.add(Flatten())
    model.add(Dense(units, activation = 'relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return model
from keras import callbacks

callbacks = [callbacks.EarlyStopping(monitor = 'val_loss', patience = 3, mode = 'auto'),
             callbacks.ModelCheckpoint('best_weights_model.h5', monitor = 'val_loss', save_best_only = True, period = 3),
             callbacks.CSVLogger('training.csv')]
model = build_model()
history = model.fit(x_train, y_train,
                    steps_per_epoch = 24,
                    epochs = 25,
                    validation_data = (x_val, y_val),
                    validation_steps = 15,
                    callbacks = callbacks)
acc = history.history['acc']
loss = history.history['loss']

val_acc = history.history['val_acc']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure(figsize = (15, 5))

plt.subplot(1,2,1)
plt.plot(epochs, acc, 'r', label = 'Training Accuracy')
plt.plot(epochs, val_acc, 'b', label = 'Validation Accuracy')
plt.title('Training vs. Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs, loss, 'r', label = 'Training Loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation Loss')
plt.title('Training vs. Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score
y_pred = model.predict_classes(x_test)

print('Test Accuracy:', accuracy_score(y_pred, y_test))
print('Recall Score:', recall_score(y_pred, y_test))
print('Precision Score:', precision_score(y_pred, y_test))
print('F1 Score:', f1_score(y_pred, y_test))