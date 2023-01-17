import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

import os
import glob

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dense, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
img = load_img("../input/intel-image-classification/seg_train/seg_train/glacier/10003.jpg")

plt.imshow(img)
plt.show()

x = img_to_array(img)
print(x.shape)
img = load_img("../input/intel-image-classification/seg_train/seg_train/street/10019.jpg")

plt.imshow(img)
plt.show()

x = img_to_array(img)
print(x.shape)
model = Sequential()

model.add(Conv2D(32,(3,3), input_shape = x.shape))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(32,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(128,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(256,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(6))
model.add(Activation("softmax"))
model.compile(loss = "categorical_crossentropy",
             optimizer = "Adam",
             metrics = ["accuracy"])
train_datagen = ImageDataGenerator(rescale=1./255,
                  shear_range=0.3,
                  horizontal_flip=True,
                  zoom_range=0.3)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    "../input/intel-image-classification/seg_train/seg_train", 
    target_size=(150, 150),
    batch_size = 32,
    color_mode="rgb",
    class_mode = "categorical")

test_generator = test_datagen.flow_from_directory(
    "../input/intel-image-classification/seg_test/seg_test", 
    target_size=(150, 150),
    batch_size = 32,
    color_mode="rgb",
    class_mode = "categorical")
batch_size = 32

hist = model.fit_generator(
    generator = train_generator,
    steps_per_epoch = 1600 // batch_size,
    epochs=20,
    validation_data=test_generator,
    validation_steps = 800 // batch_size)
print(hist.history.keys())

plt.plot(hist.history["loss"], label="Train Loss")
plt.plot(hist.history["val_loss"], label="Validation loss")
plt.legend()
plt.show()

plt.plot(hist.history["accuracy"], label="Train accuracy")
plt.plot(hist.history["val_accuracy"], label="Validation accuracy")
plt.legend()
