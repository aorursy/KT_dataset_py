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
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.models import Sequential
import keras
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
x_train = np.array(train.drop('label', axis=1))
y_train = np.array(train['label']).reshape(x_train.shape[0], 1)
x_test = np.array(test)
print(x_train.shape, y_train.shape)
print(x_test.shape)
enc = OneHotEncoder()
enc.fit(y_train)
y_train_hot = np.array(enc.transform(y_train).toarray())
print(y_train_hot.shape)
x_train, x_cv, y_train, y_cv = train_test_split(x_train, y_train_hot, test_size=0.3, random_state=420)
print(x_train.shape, y_train.shape)
print(x_cv.shape, y_cv.shape)
mean_px = x_train.mean().astype(np.float32)
std_px = x_train.std().astype(np.float32)

def standardize(x): 
    return (x-mean_px)/std_px
plt.imshow(x_train[0].reshape(28, 28))
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_cv = x_cv.reshape(x_cv.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3),input_shape = input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, kernel_size=(3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))
model.compile(loss=keras.losses.categorical_crossentropy, 
              optimizer=keras.optimizers.Adadelta(), 
              metrics=['accuracy'])
from keras.callbacks import EarlyStopping
my_callbacks = [EarlyStopping(monitor="acc", patience=5, mode=max)]
hist = model.fit(x_train, y_train, batch_size=128, 
                 epochs=100, verbose=1, validation_split=0.3, callbacks=my_callbacks)
score = model.evaluate(x_cv, y_cv)
print("Testing Loss:", score[0])
print("Testing Accuracy:", score[1])
model.summary()
score
epoch_list = list(range(1, len(hist.history['acc']) + 1))
plt.plot(epoch_list, hist.history['acc'], epoch_list, hist.history['val_acc'])
plt.legend(("Training Accuracy", "Validation Accuracy"))
plt.show()
y_pred = model.predict(x_test, batch_size=128)
y_pred.shape
y_p = y_pred.argmax(axis=1)
y_p.shape
y_p[:10]
n=3
print(y_p[n])
plt.imshow(x_test[n].reshape(28,28))
my_submission = pd.DataFrame({'ImageId': np.array(range(1,y_p.shape[0]+1)), 'Label': y_p})
my_submission.head()
#my_submission.to_csv('my_submission.csv', index=False)
from keras.layers.normalization import BatchNormalization


# Initial model

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Dropout(0.5))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
print(model.summary())
# Initial fIT & Evaluate initial model

num_epochs = 30
BatchSize = 2048

model.fit(x_train, y_train, epochs=num_epochs, batch_size=BatchSize)
test_loss, test_acc = model.evaluate(x_cv, y_cv)
print("_"*80)
print("Accuracy on test ", test_acc)
model.save_weights('mnist_model_weights.h5')
y_pred = model.predict(x_test)
y_pred[1]
y_p = y_pred.argmax(axis=1)
y_p.shape
n=3
print(y_p[n])
plt.imshow(x_test[n].reshape(28,28))
my_submission2 = pd.DataFrame({'ImageId': np.array(range(1,y_p.shape[0]+1)), 'Label': y_p})
my_submission2.head()
#my_submission2.to_csv('my_submission2.csv', index=False)
