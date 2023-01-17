# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import matplotlib.pyplot as plt
print(os.listdir("../input"))
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

# Any results you write to the current directory are saved as output.
train_im = pd.read_csv("../input/fashion-mnist_train.csv")
test_im = pd.read_csv("../input/fashion-mnist_test.csv")

#Retrieve the pixel values and the labels from the train and test set
train_img = train_im[train_im.columns[1:]]
train_lab = train_im['label']
test_img = test_im[test_im.columns[1:]]
test_lab = test_im['label']
#Convert them to numpy arrays
train_img = np.array(train_img)
test_img = np.array(test_img)

#Convert labels to categorical
train_lab = to_categorical(np.array(train_lab))
test_lab = to_categorical(np.array(test_lab))

#Reshape the values and preprocess them
img_rows, img_cols = 28,28
train_image = train_img.reshape(train_img.shape[0], img_rows, img_cols, 1)
test_image = test_img.reshape(test_img.shape[0], img_rows, img_cols, 1)
train_image = train_image.astype('float32')
test_image = test_image.astype('float32')
train_image /= 255
test_image /= 255
# p = train_img.head()
# p= p.as_matrix()
#j = train_img[0].reshape(28,28)
#print (train_image.shape)
#plt.imshow(j)
#plt.show()
input_shape = (img_rows, img_cols, 1)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd,
          loss='categorical_crossentropy',
          metrics=['accuracy'])
print (model.summary())
checkpoint = ModelCheckpoint("checkpoint.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
model.fit(train_image, train_lab,
          batch_size=50,
          epochs=10,
          verbose=1,
          validation_data=(test_image, test_lab),callbacks = [checkpoint])
score = model.evaluate(test_image, test_lab, verbose=0)
print('accuracy: ',score[1])
print('loss: ',score[0])