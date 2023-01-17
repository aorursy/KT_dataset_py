import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

import tensorflow as tf

import cv2

import random

import pickle

import time
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout,Activation, Flatten, Conv2D, MaxPool2D

from tensorflow.keras.callbacks import TensorBoard

from tensorflow.keras.preprocessing.image import ImageDataGenerator



from sklearn.model_selection import train_test_split

Datadir = '../input/kagglecatsanddogs_3367a/PetImages/'

Categories = ['Dog', 'Cat']



img_size = 100

training_data = []



for Category in Categories:

    path = os.path.join(Datadir, Category)

    print("Loading data from ", path)

    class_num = Categories.index(Category)

    for img in os.listdir(path):

        try:

            img_array = cv2.imread(os.path.join(

                path, img), cv2.IMREAD_GRAYSCALE)

            img_resize = cv2.resize(img_array, (img_size, img_size))

            training_data.append([img_resize, class_num])

        except Exception as e:

            pass
print("training_data lenght = ",len(training_data))

random.shuffle(training_data)
X, y = [], []

for feature, label in training_data:

    X.append(feature)

    y.append(label)

X = np.array(X).reshape(-1, img_size, img_size, 1)
pickle_out = open("X.pickle","wb")

pickle.dump(X,pickle_out)

pickle_out.close()



pickle_out = open("y.pickle","wb")

pickle.dump(y,pickle_out)

pickle_out.close()
pickle_in = open("X.pickle","rb")

X = pickle.load(pickle_in)



pickle_in = open("y.pickle","rb")

y = pickle.load(pickle_in)



X = tf.keras.utils.normalize(X, axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X.shape)

print(X_train.shape)

print(X_test.shape)
model = Sequential()



model.add(Conv2D(32, (3, 3), input_shape=X.shape[1:]))

model.add(Activation("relu"))

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Conv2D(32, (3, 3)))

model.add(Activation("relu"))

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Conv2D(32, (3, 3)))

model.add(Activation("relu"))

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Flatten())



model.add(Dropout(0.5))  

model.add(Dense(100))

model.add(Activation("relu"))



model.add(Dense(1))

model.add(Activation("sigmoid"))



model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])
# Use data augmentation

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True,rotation_range=20)

train_generator = train_datagen.flow(

    X_train,

    y_train,

    batch_size=32)



# do not augment validation data

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

validation_generator = test_datagen.flow(

    X_test,

    y_test,

    batch_size=32)
history = model.fit_generator(

    train_generator,

    steps_per_epoch=1000,

    epochs= 50,

    validation_data=validation_generator,

    validation_steps=150)
acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(acc) + 1)



plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()
model.save('cats_vs_dogs_with_aug.model')
def prepare(filepath):

    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    img_resize = cv2.resize(img_array, (img_size, img_size))

    return img_resize.reshape(-1,img_size,img_size,1)



trained_model = tf.keras.models.load_model('cats_vs_dogs_with_aug.model')



#prediction = trained_model.predict([prepare('your_img.jpg')])

#print(Categories[int(prediction[0][0])])