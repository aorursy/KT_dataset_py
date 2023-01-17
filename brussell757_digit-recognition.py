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
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
X_train = np.array(train_data.drop(['label'], axis = 1))
y_train = np.array(train_data['label']).reshape(-1, 1)

X_test = np.array(test_data)
print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print('X_test shape:', X_test.shape)
X_train = X_train.astype('float64')
X_train /= 255
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

X_test = X_test.astype('float64')
X_test /= 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
from keras.utils import to_categorical

y_train = to_categorical(y_train)

print('y_train shape:', y_train.shape)
from keras import models
from keras import layers
from keras import optimizers

def build_model(lr = 0.01):
    model = models.Sequential()
    model.add(layers.Conv2D(32, 3, activation = 'relu', input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[-1])))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(64, 3, activation = 'relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation = 'relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation = 'softmax'))
    model.compile(optimizer = optimizers.RMSprop(lr = lr), loss = 'categorical_crossentropy', metrics = ['acc'])
    
    return model
model = build_model(lr = 0.001)
model.summary()
from keras import callbacks

callbacks = [callbacks.EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True)]
history = model.fit(X_train, y_train,
                    epochs = 25,
                    batch_size = 64,
                    validation_split = 0.2,
                    callbacks = callbacks)
import matplotlib.pyplot as plt

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(1, len(acc) + 1)
    
    plt.figure(figsize = (25, 5))
    
    plt.subplot(1,2,1)
    plt.plot(epochs, acc, c = 'b', label = 'train acc')
    plt.plot(epochs, val_acc, c = 'g', label = 'val acc')
    plt.title('Train vs Val Acc')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(epochs, loss, c = 'b', label = 'train loss')
    plt.plot(epochs, val_loss, c = 'g', label = 'val loss')
    plt.title('Train vs Val Loss')
    plt.legend()
    
    plt.show()
plot_history(history)
predictions = model.predict(X_test)
predictions = predictions.argmax(axis = 1)
image_ids = pd.DataFrame(np.arange(1, len(X_test) + 1))
preds = pd.DataFrame(predictions)

output = pd.DataFrame(np.concatenate([image_ids, preds], axis = 1))
output.columns = ['ImageId', 'Label']

output.to_csv('output.csv', encoding = 'utf-8')
