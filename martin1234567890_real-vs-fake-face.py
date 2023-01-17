!pip install tensorflow-gpu==2.0.0-alpha

!pip install split-folders
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import GlobalAveragePooling2D

from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.applications import MobileNetV2

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Dropout

from tensorflow.keras.layers import Dense

from tensorflow.keras import Sequential

import matplotlib.pyplot as plt

import tensorflow as tf

import split_folders

import numpy as np

import cv2

import os





print(os.listdir("../input"))
real = "../input/real_and_fake_face_detection/real_and_fake_face/training_real/"

fake = "../input/real_and_fake_face_detection/real_and_fake_face/training_fake/"



real_path = os.listdir(real)

fake_path = os.listdir(fake)
def load_img(path):

    image = cv2.imread(path)

    image = cv2.resize(image, (224, 224))

    return image[...,::-1]
fig = plt.figure(figsize=(4, 4))



for i in range(16):

    plt.subplot(4, 4, i+1)

    plt.imshow(load_img(real + real_path[i]), cmap='gray')

    plt.title("real face")

    plt.axis('off')



plt.show()
fig = plt.figure(figsize=(4, 4))



for i in range(16):

    plt.subplot(4, 4, i+1)

    plt.imshow(load_img(fake + fake_path[i]), cmap='gray')

    plt.title("fake face")

    plt.axis('off')



plt.show()
train_datagen = ImageDataGenerator(horizontal_flip=True,

                                   vertical_flip=False,

                                   rescale=1./255,

                                   )
dataset_path = "../input/real_and_fake_face_detection/real_and_fake_face"
train = train_datagen.flow_from_directory(dataset_path,

                                          class_mode="binary",

                                          target_size=(96, 96),

                                          batch_size=32)
mobilenetV2 = MobileNetV2(input_shape=(96, 96, 3),

                          include_top=False,

                          weights='imagenet'

                          )



average_layer = GlobalAveragePooling2D()



model = Sequential([

    mobilenetV2,

    average_layer,

    Dense(256, activation=tf.nn.relu),

    BatchNormalization(),

    Dropout(0.2),

    Dense(2, activation=tf.nn.softmax)

])
model.compile(optimizer=Adam(lr=0.001),

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
def scheduler(epoch):

    if epoch <= 2:

        return 0.001

    elif epoch > 2 and epoch <= 15:

        return 0.0001 

    else:

        return 0.00001



lr_callbacks = tf.keras.callbacks.LearningRateScheduler(scheduler)
model.fit_generator(train,

                    epochs=50,

                    callbacks=[lr_callbacks])
model.evaluate_generator(train)
model.save("model.h5")