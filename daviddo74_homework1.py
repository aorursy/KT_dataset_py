# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import cv2

import keras

from keras.preprocessing.image import ImageDataGenerator



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def GetMoreData(img):

    datagen = ImageDataGenerator(

        rotation_range=10,

        width_shift_range=0.1,

        height_shift_range=0.1,

        shear_range=0.15,

        zoom_range=0.1,

        channel_shift_range=10,

        horizontal_flip=True,

    )

    image = np.expand_dims(img,0)

    aug_iter = datagen.flow(image)

    aug_images = [next(aug_iter)[0].astype(np.int) for i in range(10)]

    return aug_images
DATA_DIR_TRAIN = "../input/catndog/catndog/train"

CATEGORIES = ["cat", "dog"]

x_train = []

y_train = []

IMG_SIZE = 32

for category in CATEGORIES:

    path = os.path.join(DATA_DIR_TRAIN, category)

    class_num = CATEGORIES.index(category)

    for img_path in os.listdir(path):

        img = cv2.imread(os.path.join(path, img_path))

        new_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        x_train.append(new_img)

        y_train.append(class_num)

        new_data = GetMoreData(new_img)

        for i in range(10):

            x_train.append(new_data[i])

            y_train.append(class_num)

        
DATA_DIR_TEST = "../input/catndog/catndog/test"

x_test = []

y_test = []

IMG_SIZE = 32

for category in CATEGORIES:

    path = os.path.join(DATA_DIR_TEST, category)

    class_num = CATEGORIES.index(category)

    for img_path in os.listdir(path):

        img = cv2.imread(os.path.join(path, img_path))

        new_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        x_test.append(new_img)

        y_test.append(class_num)
for i in range(0, 9):

  ax = plt.subplot(3, 3, i+1)

  ax = plt.imshow(x_train[i])

  plt.title(y_train[i])

  plt.xticks([])

  plt.yticks([])
train_label_dict = dict()



for i in range(0,2):

  n_samples_train = y_train.count(i)

  train_label_dict[i] = n_samples_train

    

train_label_dict
x_train_np, y_train_np = np.array(x_train), np.array(y_train)
x_test_np, y_test_np = np.array(x_test), np.array(y_test)
scaled_x_train, scaled_x_test = x_train_np / 255.0, x_test_np / 255.0
model = keras.Sequential([

    keras.layers.Conv2D(filters=32, kernel_size=(3,3),activation='relu'),

    keras.layers.MaxPool2D(pool_size=(2, 2)),

    keras.layers.Conv2D(filters=64, kernel_size=(3,3),activation='relu'),

    keras.layers.MaxPool2D(pool_size=(2, 2)),

    keras.layers.Conv2D(filters=128, kernel_size=(3,3),activation='relu'),

    keras.layers.MaxPool2D(pool_size=(2, 2)),

    keras.layers.Dropout(0.8),

    keras.layers.Flatten(),

    keras.layers.Dense(128, activation = 'relu'),

    keras.layers.Dense(1, activation='sigmoid')

])



#model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics=['accuracy'])



model.fit(scaled_x_train, y_train_np, epochs=10)
loss, accuracy_score = model.evaluate(scaled_x_test, y_test_np)

print('test accuracy: ', accuracy_score)

print('test loss: ', loss)
random_array = np.random.randint(85,size=9)

random_array
for i in range(0, 9):

  ax = plt.subplot(3, 3, i+1)  

  ax = plt.imshow(x_test[random_array[i]])

  prediction = np.argmax(model.predict(x_test_np[random_array[i]].reshape(1, 32, 32, 3)), axis=1)

  plt.title(prediction)

  plt.xticks([])

  plt.yticks([])