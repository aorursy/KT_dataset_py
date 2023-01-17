!ls ../input/fruits-360_dataset/fruits-360
import os



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from PIL import Image

import tensorflow as tf

from sklearn.model_selection import train_test_split
DATA_PATH = '../input/fruits-360_dataset/fruits-360/'



def read_data(path, classes):

    images, class_names = [], []

    for class_name in classes:

        for img_name in os.listdir(path + class_name):

            img = Image.open(f'{path}{class_name}/{img_name}')

            img = img.resize((100, 100))

            images.append(np.array(img))

            class_names.append(class_name)

    return np.asarray(images), np.asarray(class_names)
classes = ["Apple Golden 1", "Apple Golden 2", "Apple Golden 3"

           ,"Avocado","Banana","Cherry 1","Cocos","Kiwi"

           ,"Lemon","Mango", "Orange", "Apple Red 1", "Apple Red 2", "Apple Red 3"]

num_classes = len(classes)
X_train, y_train = read_data(DATA_PATH+'Training/', classes)

X_test, y_test = read_data(DATA_PATH+'Test/', classes)
for i in range(len(y_train)):

    y_train[i] = classes.index(y_train[i])

for i in range(len(y_test)):

    y_test[i] = classes.index(y_test[i])
X_train = X_train.astype(np.float32)

X_train /= 255.



X_test = X_test.astype(np.float32)

X_test /= 255.
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout

from keras.models import Sequential

from keras.preprocessing.image import ImageDataGenerator



model = Sequential()



model.add(Conv2D(input_shape=X_train.shape[1:], filters=16, kernel_size=2, padding="same", activation='relu'))

model.add(MaxPool2D(pool_size=2))



model.add(Conv2D(filters=32, kernel_size=2, padding="same", activation='relu'))

model.add(MaxPool2D(pool_size=2))



model.add(Conv2D(filters=64, kernel_size=2, padding="same", activation='relu'))

model.add(MaxPool2D(pool_size=2))



model.add(Conv2D(filters=128, kernel_size=2, padding="same", activation='relu'))

model.add(MaxPool2D(pool_size=2))



model.add(Flatten())

model.add(Dropout(0.4))



model.add(Dense(128))

model.add(Dense(num_classes, activation='softmax'))



model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
hist = model.fit(X_train, y_train, batch_size=64, validation_data=(X_test, y_test), epochs=20)
plt.plot(hist.history['val_loss'])

plt.plot(hist.history['loss'])
index = [np.random.randint(0, len(y_test)) for i in range(4)]

plt.figure(figsize=(16, 8))

for i in range(4):

    plt.subplot(1, 4, i+1)

    plt.imshow(X_train[index[i]])

    plt.title(classes[np.argmax(model.predict(np.array(X_train[index[i]]).reshape(-1, 100, 100, 3)), axis=1)[0]])