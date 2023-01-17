import os

import cv2

import argparse

import numpy as np

import random

import matplotlib.pyplot as plt

from keras import backend as K

from keras.optimizers import Adam

from keras.regularizers import l2

from keras.applications import VGG16

from keras.models import Sequential

from keras.layers.core import Flatten

from keras.layers.core import Dropout

from keras.layers.core import Dense

from keras.layers.core import Activation

from keras.layers.convolutional import Conv2D

from keras.layers.normalization import BatchNormalization

from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from keras.preprocessing.image import ImageDataGenerator
(height, width, depth) = (200, 200, 3)

classes=102
#Loading the Data

path = '../input/caltech101/Caltech101/'



x_train = []

y_train = []

x_test = []

y_test = []



for category in os.listdir(path):

    counter = 0

    images_in_category = os.listdir(os.path.join(path, category))

    #Shuffle images in each category

    random.shuffle(images_in_category)

    for image_name in images_in_category:

        fullpath = os.path.join(path, category, image_name)        

        label = fullpath.split(os.path.sep)[-2]

        image = cv2.imread(fullpath)

        image = cv2.resize(image, (height, width))

        if (counter < 30):      

            x_train.append(image)

            y_train.append(label)

        else:

            x_test.append(image)

            y_test.append(label)

        counter = counter + 1
# Format dataset

x_train = np.array(x_train, dtype="float") / 255.0

x_test = np.array(x_test, dtype="float") / 255.0

lb = LabelBinarizer()

y_train = lb.fit_transform(y_train)

y_test = lb.fit_transform(y_test)
print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)
# Build pretrained VGG16 model

model = Sequential()

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape = (height, width, depth))

for layer in vgg16.layers:

    layer.trainable = False

model = Sequential()

model.add(vgg16)

# FC layers

model.add(Flatten())

model.add(Dense(256, activation='relu', name = 'fc1'))

model.add(Dense(256, activation='relu', name = 'fc2'))

model.add(Dropout(0.5))

model.add(Dense(classes, activation='softmax',  name = 'fc3'))
model.summary()
# Compile model

model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=1e-4, decay=1e-4 / 50), metrics=["accuracy"])
#Augmenting

aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,

    width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,

    horizontal_flip=True, fill_mode="nearest")
# Train the model for 200 epochs

epochs = 200

hist = model.fit_generator(aug.flow(x_train, y_train, batch_size=64),

    validation_data=(x_test, y_test), steps_per_epoch=len(x_train) // 64,

    epochs=epochs,verbose=2)
# Evaluate the model

scores = model.evaluate(x_test, y_test, verbose=2)

print("Training Accuracy: %.2f%%" % (hist.history['acc'][epochs-1]*100))

print("Testing Accuracy: %.2f%%" % (scores[1]*100))
# Plot the graphs between loss vs value loss and accuracy vs value accuracy

plt.plot(hist.history['loss'])

plt.plot(hist.history['val_loss'])

plt.legend(['train','test'])

plt.title('loss')

plt.savefig("loss.png",dpi=300,format="png")

plt.figure()

plt.plot(hist.history['acc'])

plt.plot(hist.history['val_acc'])

plt.legend(['train','test'])

plt.title('accuracy')

plt.savefig("accuracy.png",dpi=300,format="png")