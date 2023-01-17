import numpy as np 

import pandas as pd 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from keras import layers

from keras import models

from keras.utils import to_categorical

from keras.applications import VGG16

from keras.datasets import cifar10

from keras.preprocessing import image

import matplotlib.pyplot as plt

import cv2
def graph(history):

    epoklar = range(1,len(history["loss"])+1)

    

    plt.plot(epoklar,history["loss"],label="Training Loss")

    plt.plot(epoklar,history["val_loss"],label="Validation Loss")

    plt.title("Loss")

    plt.legend()

    plt.show()

    

    plt.plot(epoklar,history["acc"],label="Training Accuracy")

    plt.plot(epoklar,history["val_acc"],label="Validation Accuracy")

    plt.title("Accuracy")

    plt.legend()

    plt.show()
(train_images,train_labels),(test_images,test_labels) = cifar10.load_data()
def resize(images):

    yeni = np.zeros((images.shape[0],96,96,3),dtype=np.float32)

    for i in range(len(images)):

        yeni[i] = cv2.resize(images[i,:,:,:],(96,96))

    return yeni
train_images = resize(train_images)

test_images = resize(test_images)
train_images.shape
train_labels.shape
numberOfClass = len(np.unique(train_labels))

numberOfClass
plt.imshow(test_images[20].astype(np.uint8))

plt.axis("off")

plt.show()
#one-hot encoding

train_labels = to_categorical(train_labels)

test_labels = to_categorical(test_labels)
x_train = train_images[:35000]

y_train = train_labels[:35000]



x_valid = train_images[35000:]

y_valid = train_labels[35000:]



x_test = test_images

y_test = test_labels
#data augmentation and normalize images with using ImageDataGenerator

train_datagen = image.ImageDataGenerator(

      rescale=1./255,

      rotation_range=30,

      width_shift_range=0.15,

      height_shift_range=0.15,

      shear_range=0.15,

      zoom_range=0.1,

      horizontal_flip=True,

      fill_mode='nearest')



valid_datagen = image.ImageDataGenerator(rescale=1./255)

test_datagen = image.ImageDataGenerator(rescale=1./255)
batch_size = 256

epoch = 30

size = x_train.shape[1:]
train_generator = train_datagen.flow(x_train,y_train,batch_size=batch_size)

valid_generator = valid_datagen.flow(x_valid,y_valid,batch_size=batch_size)

test_generator = test_datagen.flow(x_test,y_test,batch_size=batch_size)
vgg16 = VGG16(weights="imagenet",include_top=False,input_shape=size)
vgg16.summary()
#create new model and add pretrained model in this model

model = models.Sequential()

for i in range(len(vgg16.layers)):

    model.add(vgg16.layers[i])



for i in range(len(model.layers)): #freeze pretrained model

    model.layers[i].trainable = False



#add new layers your model

model.add(layers.Flatten())

model.add(layers.Dropout(0.40))

model.add(layers.Dense(256,activation="relu"))

model.add(layers.Dense(numberOfClass,activation="softmax"))
model.compile(optimizer="rmsprop",loss="categorical_crossentropy",

              metrics=["acc"])
model.summary()
(4608*256 + 256) + (256*10 + 10)
history = model.fit_generator(generator=train_generator,

                steps_per_epoch=len(train_generator),epochs=epoch,

                validation_data=valid_generator,

                validation_steps=len(valid_generator))
graph(history.history)
loss,acc = model.evaluate_generator(test_generator,

                        steps=len(test_generator))
print("Test Accuracy =",acc)
model.summary()
#select after blocks block5_conv1 to fine tunnig

trnable = False

for i in model.layers:

    if i.name == "block5_conv1":

        trnable = True

    i.trainable = trnable
from keras import optimizers
#compile model with very low learning rate

model.compile(optimizer=optimizers.RMSprop(lr=1e-5),

        loss="categorical_crossentropy",metrics=["acc"])
model.summary()
batch_size = 256

epoch = 50
history = model.fit_generator(generator=train_generator,

                steps_per_epoch=len(train_generator),epochs=epoch,

                validation_data=valid_generator,

                validation_steps=len(valid_generator))
graph(history.history)
loss,acc = model.evaluate_generator(test_generator,

                            steps=len(test_generator))
print("Test Accuracy =",acc)