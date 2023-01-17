import pandas as pd

import numpy as np



from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing import image

from keras.callbacks import EarlyStopping,ModelCheckpoint

from keras.models import Sequential,Model

from keras.layers import Conv2D,MaxPooling2D

from keras.layers import Activation,Dropout,Dense,Flatten

from keras import optimizers

from keras.applications import VGG16



import cv2

import os

import random 

import itertools

from glob import iglob

from collections import Counter

from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt
BASE_DATASET_FOLDER = '/kaggle/input/diabetic-retinopathy-detection/data'

VALIDATION_FOLDER = 'validation'

TEST_FOLDER = 'test'

TRAIN_FOLDER = 'training'
def plotImages(images_arr):

    fig, axes = plt.subplots(1, 5, figsize=(20,20))

    axes = axes.flatten()

    for img, ax in zip( images_arr, axes):

        ax.imshow(img)

        ax.axis('off')

    plt.tight_layout()

    plt.show()
IMAGE_SIZE = (224,224)

INPUT_SHAPE = (224,224, 3)
TRAIN_BATCH_SIZE =80

VAL_BATCH_SIZE = 15

EPOCHS = 50

LEARNING_RATE = 0.0001
train_datagen=ImageDataGenerator(

    

    rescale=1./255,

    #featurewise_center=False,

    #samplewise_center=False,

    #featurewise_std_normalization=False,

    #samplewise_std_normalization=False,

    #rotation_range=45,

    #zoom_range=0.2,

    width_shift_range=0.2,

    height_shift_range=0.2,

    horizontal_flip=True,

    #vertical_flip=True,

    fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(

        os.path.join(BASE_DATASET_FOLDER, TRAIN_FOLDER),

        target_size=IMAGE_SIZE,

        batch_size=TRAIN_BATCH_SIZE,

        class_mode='categorical', 

        shuffle=True)
augmented_images = [train_generator[0][0][0] for i in range(5)]

plotImages(augmented_images)
val_datagen=ImageDataGenerator(rescale=1./255)

val_generator=val_datagen.flow_from_directory(

    os.path.join(BASE_DATASET_FOLDER, VALIDATION_FOLDER),

    target_size=IMAGE_SIZE,

    class_mode='categorical', 

    shuffle=False)
test_datagen=ImageDataGenerator(rescale=1./255)

test_generator=test_datagen.flow_from_directory(

    os.path.join(BASE_DATASET_FOLDER, TEST_FOLDER),

    target_size=IMAGE_SIZE,

    class_mode='categorical', 

    shuffle=False)
classes = {v: k for k, v in train_generator.class_indices.items()}

print(classes)
CATEGORIES = ["0", "1","2","3",'4']

for category in CATEGORIES:  

    path = os.path.join('/kaggle/input/diabetic-retinopathy-detection/data/training',category)  

    x=0

    for img in os.listdir(path): 

        x+=1

        img_array = cv2.imread(os.path.join(path,img))  

        plt.imshow(img_array)  #r graph it

        plt.show()  # display!

        if x==10 : 

            break
vgg_model = VGG16( include_top=False, input_shape=INPUT_SHAPE)

for layer in vgg_model.layers[:-4]:

    layer.trainable = False
model=Sequential()

""""model.add(Conv2D(32, (3,3), activation='relu', input_shape=INPUT_SHAPE))

model.add(Conv2D(64, (3,3), activation='relu'))

model.add(Conv2D(64, (3,3), activation='relu'))

model.add(MaxPooling2D(2, 2))

model.add(Conv2D(64, (3,3), activation='relu'))

model.add(Conv2D(64, (3,3), activation='relu'))

model.add(MaxPooling2D(2, 2))

model.add(Dropout(.2))

model.add(Conv2D(128, (3,3), activation='relu'))

model.add(MaxPooling2D(2, 2))

model.add(Dropout(.2))

model.add(Conv2D(256, (3,3), activation='relu'))

model.add(Conv2D(512, (3,3), activation='relu'))

model.add(Conv2D(512, (3,3), activation='relu'))

model.add(MaxPooling2D(2,2))

model.add(Dropout(.2))"""

model.add(vgg_model)

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(.2))

model.add(Dense(128, activation='relu'))

model.add(Dropout(.3))

model.add(Dense(len(classes), activation='softmax'))
model.summary()

#sgd=optimizers.SGD(lr=LEARNING_RATE,decay=1e-6,momentum=0.9)
model.compile(loss='categorical_crossentropy',

              optimizer=optimizers.Adam(LEARNING_RATE),

              metrics=['acc'])
es = EarlyStopping(

    monitor='val_acc', 

    mode='max',

    patience=6

)
train_generator.samples//train_generator.batch_size

#val_generator.samples//val_generator.batch_size
history = model.fit_generator(

        train_generator,

        steps_per_epoch=train_generator.samples//train_generator.batch_size,

        epochs=EPOCHS,

        validation_data=val_generator,

        validation_steps=val_generator.samples//val_generator.batch_size,

        callbacks=[es]

)
plt.figure(figsize=[14,10])

plt.subplot(211)

plt.plot(history.history['loss'],'r',linewidth=3.0)

plt.plot(history.history['val_loss'],'b',linewidth=3.0)

plt.legend(['Training loss', 'Validation Loss'],fontsize=18)

plt.xlabel('Epochs ',fontsize=16)

plt.ylabel('Loss',fontsize=16)

plt.title('Loss Curves',fontsize=16)

 

# Accuracy Curves

plt.figure(figsize=[14,10])

plt.subplot(212)

plt.plot(history.history['acc'],'r',linewidth=3.0)

plt.plot(history.history['val_acc'],'b',linewidth=3.0)

plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)

plt.xlabel('Epochs ',fontsize=16)

plt.ylabel('Accuracy',fontsize=16)

plt.title('Accuracy Curves',fontsize=16)
loss,accuracy=model.evaluate_generator(test_generator,steps=test_generator.samples//test_generator.batch_size)

print("accuracy :%f \n loss: %f"%(accuracy,loss))