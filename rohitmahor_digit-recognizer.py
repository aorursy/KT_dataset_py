# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dataset = pd.read_csv('../input/train.csv')
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0:1].values

test_set = pd.read_csv('../input/test.csv').values
print(dataset.info())
print(X.shape)
print(y.shape)
X = X.reshape(X.shape[0], 28, 28, 1)
test_set = test_set.reshape(test_set.shape[0], 28, 28, 1)
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
encoder.fit(y)
y = encoder.transform(y).toarray()
print(y.shape)
X = X/255.0
test_set = test_set/255.0
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
# intialization
clf = Sequential()

# Add layers
clf.add(Convolution2D(filters=32, kernel_size=(5,5), input_shape=(28, 28, 1), activation='relu'))
clf.add(Convolution2D(filters=32, kernel_size=(5, 5), activation='relu'))
clf.add(MaxPooling2D(2,2))
clf.add(Dropout(0.25))

clf.add(Convolution2D(filters=32, kernel_size=(3, 3), activation='relu'))
clf.add(Convolution2D(filters=32, kernel_size=(3, 3), activation='relu'))
clf.add(MaxPooling2D(2,2))
clf.add(Dropout(0.25))

clf.add(Flatten())

# full connection
clf.add(Dense(output_dim=256, activation='relu'))
clf.add(Dropout(0.25))
clf.add(Dense(output_dim=10, activation='sigmoid'))

# Compile the model
clf.compile(optimizer = 'adam' , loss = "categorical_crossentropy", metrics=["accuracy"])
datagen = ImageDataGenerator(rotation_range=15,  # Rotate by 15 degree
    width_shift_range=0.1, # width shift by 10%
    height_shift_range=0.1, # height shift by 10%
    zoom_range=0.1,  # Zoom by 10%
    preprocessing_function=None)
datagen.fit(X_train)
from keras.callbacks import ReduceLROnPlateau
learningRate = ReduceLROnPlateau(monitor='val_loss',
                                 factor=0.5,
                                 patience=5,
                                 verbose=0,
                                 min_lr=0.00001)
history = clf.fit_generator(datagen.flow(X_train,y_train, batch_size=100),
                              epochs = 30, validation_data = (X_test,y_test),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // 100
                              , callbacks=[learningRate])
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
y_pred = clf.predict(X_test)
y_pred1 = [w.argmax() for w in y_pred]
y_test1 = [w.argmax() for w in y_test]

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test1, y_pred1)
print(cm)

score = clf.evaluate(X_test, y_test)
print(score)
# predict final results
results = clf.predict(test_set)

# select the index with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_result_1.csv",index=False)