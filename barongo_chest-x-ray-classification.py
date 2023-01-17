from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os # used for navigating to image path

import imageio # used for writing images

import tensorflow as tf





from PIL import Image

from skimage.io import imread

from skimage.transform import resize

from pathlib import Path



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import the data directory

dataPath = Path('../input/chest-xray-pneumonia/chest_xray/chest_xray')



# we define the path to train directory using fancy pathlib

trainPath = dataPath / 'train'



# same with validatio data directory using pathlib

valPath = dataPath / 'val'



# Again we do the same with 

testPath = dataPath / 'test'
#defining the test dataset with pnuemonia xray

trainPathPneum = dataPath / 'train/PNEUMONIA/'



#defining the test dataset with normal xray

trainPathNorm = dataPath / 'train/NORMAL/'
#defining the test dataset with pnuemonia xray

ValPathPneum = dataPath / 'val/PNEUMONIA/'



#defining the test dataset with normal xray

ValPathNorm = dataPath / 'val/NORMAL/'
#put get all the training images and put them into a list

#Pneumonia Directory

pneumXray = trainPathNorm.glob('*.jpeg')

#Pneumonia test directory

normalXray = trainPathPneum.glob('*.jpeg')

#create and empty list

trainData = []

# Iterating through all the Pneumonia cases. We assign a label of 1

for img in pneumXray:

    trainData.append((img, 1))

# Iterating through all the Normal cases. We assign a label of 0

for img in normalXray:

    trainData.append((img, 0))
#check if the images have been appened correctly - checking the first image on the list

print(trainData[0][0])
#display the first image

image1 = trainData[0][0]

f, ax = plt.subplots(1, figsize=(8,4))



img = imread(image1)

ax.imshow(img, cmap='gray')

ax.axis('off')

plt.show()
PneumTrain = len(os.listdir(trainPathPneum))

NormTrain = len(os.listdir(trainPathNorm))



PneumVal = len(os.listdir(ValPathPneum))

NormVal = len(os.listdir(ValPathNorm))



allTrain = PneumTrain + NormTrain

allVal = PneumVal + NormVal
print(trainPathPneum)
#convert the images into apandas  data frame 

trainData = pd.DataFrame(trainData, columns=['image', 'label'],index=None)

#shuffling the df

trainData = trainData.sample(frac=1.).reset_index(drop=True)

#let's view the dataframe

trainData.head(3)
allTrain_y = np.asarray(allTrain)

allVal_y = np.asarray(allVal)
#how many of each class do we have

count = trainData['label'].value_counts()

print(count)
#let us display 10 random images and assign lables to them



#change train Data to list the concatenate the two samples



#sample images from both classes

pneumS = (trainData[trainData['label']==1]['image'].iloc[:9]).tolist()

normS = (trainData[trainData['label']==0]['image'].iloc[:9]).tolist()



#concatenation of the two random samples

allSamples = pneumS + normS

del pneumS, normS



#display sample images

f, ax = plt.subplots(2,5, figsize=(30,10))

for i in range(10):

    img = imread(allSamples[i])

    ax[i//5, i%5].imshow(img, cmap='gray')

    if i<5:

        ax[i//5, i%5].set_title("Pneumonia")

    else:

        ax[i//5, i%5].set_title("Normal")

    ax[i//5, i%5].axis('off')

    ax[i//5, i%5].set_aspect('auto')

plt.show()
batch_size = 128

epochs = 15

IMG_HEIGHT = 150

IMG_WIDTH = 150
#Normalization of the data - treating all values as between 0 and 1

trainImageGen = ImageDataGenerator(rescale=1./255) # Generator for our training data

valImageGen = ImageDataGenerator(rescale=1./255) # Generator for our validation data
train_data_gen = trainImageGen.flow_from_directory(batch_size=batch_size,

                                                           directory=trainPath,

                                                           shuffle=True,

                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),

                                                           class_mode='binary')
val_data_gen = valImageGen.flow_from_directory(batch_size=batch_size,

                                                           directory=valPath,

                                                           shuffle=True,

                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),

                                                           class_mode='binary')
model = Sequential([

    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),

    MaxPooling2D(),

    Conv2D(32, 3, padding='same', activation='relu'),

    MaxPooling2D(),

    Conv2D(64, 3, padding='same', activation='relu'),

    MaxPooling2D(),

    Flatten(),

    Dense(512, activation='relu'),

    Dense(1)

])
model.compile(optimizer='adam',

              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),

              metrics=['accuracy'])
#model Summary

model.summary()
#training the model

history = model.fit_generator(

    train_data_gen,

    steps_per_epoch=allTrain_y // batch_size,

    epochs=epochs,

    validation_data=val_data_gen,

    validation_steps=allVal_y // batch_size

)
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



loss=history.history['loss']

val_loss=history.history['val_loss']



epochs_range = range(epochs)



plt.figure(figsize=(8, 8))

plt.subplot(1, 2, 1)

plt.plot(epochs_range, acc, label='Training Accuracy')

plt.plot(epochs_range, val_acc, label='Validation Accuracy')

plt.legend(loc='lower right')

plt.title('Training and Validation Accuracy')



plt.subplot(1, 2, 2)

plt.plot(epochs_range, loss, label='Training Loss')

plt.plot(epochs_range, val_loss, label='Validation Loss')

plt.legend(loc='upper right')

plt.title('Training and Validation Loss')

plt.show()
#Dropout

model_new = Sequential([

    Conv2D(16, 3, padding='same', activation='relu', 

           input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),

    MaxPooling2D(),

    Dropout(0.2),

    Conv2D(32, 3, padding='same', activation='relu'),

    MaxPooling2D(),

    Conv2D(64, 3, padding='same', activation='relu'),

    MaxPooling2D(),

    Dropout(0.2),

    Flatten(),

    Dense(512, activation='relu'),

    Dense(1)

])
#compile the new model

model_new.compile(optimizer='adam',

                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),

                  metrics=['accuracy'])



model_new.summary()
#train the new model

history = model_new.fit_generator(

    train_data_gen,

    steps_per_epoch=allTrain_y // batch_size,

    epochs=epochs,

    validation_data=val_data_gen,

    validation_steps=allVal_y // batch_size

)
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



loss = history.history['loss']

val_loss = history.history['val_loss']



epochs_range = range(epochs)



plt.figure(figsize=(8, 8))

plt.subplot(1, 2, 1)

plt.plot(epochs_range, acc, label='Training Accuracy')

plt.plot(epochs_range, val_acc, label='Validation Accuracy')

plt.legend(loc='lower right')

plt.title('Training and Validation Accuracy')



plt.subplot(1, 2, 2)

plt.plot(epochs_range, loss, label='Training Loss')

plt.plot(epochs_range, val_loss, label='Validation Loss')

plt.legend(loc='upper right')

plt.title('Training and Validation Loss')

plt.show()