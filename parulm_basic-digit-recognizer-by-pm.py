# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split
#from tensorflow.python import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

from keras.datasets import mnist
img_rows, img_cols = 28, 28
num_classes = 10

def data_prep(raw):
    out_y = to_categorical(raw.label, num_classes)
    print(type(out_y))

    num_images = raw.shape[0]
    x_as_array = raw.values[:,1:]
    x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1)
    out_x = x_shaped_array / 255
    return out_x, out_y

train_file = "../input/train.csv"
raw_data = pd.read_csv(train_file)

kaggle_x, kaggle_y = data_prep(raw_data)
kag_arr_x = np.array(kaggle_x).reshape(42000, 28, 28, 1)
print(type(kaggle_y))

#X_train, X_val, Y_train, Y_val = train_test_split(x, y, test_size = 0.1)
#raw_data.label
# Load Data from MNIST

(train_imagesRaw, train_labelsRaw), (test_imagesRaw, test_labelsRaw) = mnist.load_data()
# Prepare the MNIST data

#mnist_x, mnist_y = data_prep(train_imagesRaw)
print(train_imagesRaw.shape)
print(kag_arr_x.shape)

train_imagesKeras = train_imagesRaw.copy()
train_labelsKeras = train_labelsRaw.copy()
train_imagesKeras = train_imagesKeras.reshape(60000,28,28,1)
train_imagesKeras = train_imagesKeras.astype('float32') / 255
print("train_imagesKeras ",train_imagesKeras.shape)
print("train_labelsKeras ",train_labelsKeras.shape)

train_labelsKeras = to_categorical(train_labelsKeras)
print("train_labelsKeras ",train_labelsKeras.shape)
#Concatenate the two datasets
train_images = np.concatenate((train_imagesKeras,kag_arr_x), axis=0)
print("new Concatenated train_images ", train_images.shape)
print("_"*50)

train_labels = np.concatenate((train_labelsKeras,kaggle_y), axis=0)
print("new Concatenated train_labels ", train_labels.shape)

#split the big fat data

X_train, X_val, Y_train, Y_val = train_test_split(train_images, train_labels, test_size = 0.2)
model = Sequential()
model.add(Conv2D(32, kernel_size=(6, 6),
                 strides=2,
                 activation='relu',
                 padding='same',
                 input_shape=(img_rows, img_cols, 1)))
model.add(Conv2D(32, kernel_size=(6, 6), strides=2, activation='relu', padding='same'))
#model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

#model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same'))
#model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same'))
#model.add(MaxPool2D(pool_size=(2,2)))
#model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))

#optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(loss="categorical_crossentropy",
              optimizer='adam',
              metrics=['accuracy'])

learning_rate_reduction = ReduceLROnPlateau()

print(model.summary())
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(X_train)
batch_size = 100
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = 30, validation_data = (X_val,Y_val),
                              steps_per_epoch=X_train.shape[0] // batch_size ,
                              callbacks=[learning_rate_reduction])

#skipping data augmentation
#model.fit(x, y, batch_size=128, epochs=4, validation_split = 0.2)
test_file = "../input/test.csv"
test_data = pd.read_csv(test_file)

def test_prep(tdata):
    num_images = tdata.shape[0]
    x_as_array = tdata.values[:,:]
    x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1)
    out_x = x_shaped_array / 255
    return out_x

test_X = test_prep(test_data)
ans = model.predict_classes(test_X)
results = pd.Series(ans,name="Label")
#random code block for testing
#test_data.isnull().any().describe()
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("output_pm.csv",index=False)
