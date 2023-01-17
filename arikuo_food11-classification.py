import os

print(os.listdir('../input/food11-image-dataset'))
train_dir = '../input/food11-image-dataset/training'

valid_dir = '../input/food11-image-dataset/validation'

test_dir  = '../input/food11-image-dataset/evaluation'
dirNames = os.listdir(train_dir)



for dirName in dirNames:

    if '.txt' in dirName:

        print(dirName)

        dirNames.remove(dirName)





print(dirNames)

print(len(dirNames))
labels = dirNames.sort()
trainFiles = []

trainClasses = []



for dirName in dirNames:

    for file in os.listdir(train_dir+"/"+dirName):

        trainFiles.append(train_dir+"/"+dirName+"/"+file)

        trainClasses.append(dirName)



print(len(trainFiles), len(trainClasses))
import matplotlib.pyplot as plt

import matplotlib.image as mpimg



plt.imshow(mpimg.imread(trainFiles[0]))
from collections import Counter

import seaborn as sns

sns.set_style("whitegrid")



def plot_equilibre(categories, counts):



    plt.figure(figsize=(12, 8))



    sns_bar = sns.barplot(x=categories, y=counts)

    sns_bar.set_xticklabels(categories, rotation=45)

    plt.title('Equilibre of Training Dataset')

    plt.show()
categories = dirNames

counts = []

[counts.append(trainClasses.count(dirName)) for dirName in dirNames]



plot_equilibre(categories, counts)
import numpy as np

import cv2

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.preprocessing.image import ImageDataGenerator
target_size=(224,224)

batch_size = 16
train_datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=30,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.15,

    zoom_range=0.15,

    horizontal_flip=True)



train_generator = train_datagen.flow_from_directory(

    train_dir,

    target_size=target_size,

    batch_size=batch_size,

    color_mode='rgb',    

    shuffle=True,

    seed=42,

    class_mode='categorical')
valid_datagen = ImageDataGenerator(rescale=1./255)



valid_generator = valid_datagen.flow_from_directory(

    valid_dir,

    target_size=target_size,

    batch_size=batch_size,

    color_mode='rgb',

    shuffle=False,    

    class_mode='categorical')
test_datagen = ImageDataGenerator(rescale=1./255)



test_generator = test_datagen.flow_from_directory(

    test_dir,

    target_size=target_size,

    batch_size=batch_size,

    color_mode='rgb',

    shuffle=False,    

    class_mode='categorical')
import tensorflow as tf

import tensorflow.keras as keras

#from tensorflow.keras.applications import EfficientNetB7

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D

from tensorflow.keras.layers import Input, BatchNormalization, Activation, LeakyReLU, Concatenate

from tensorflow.keras.regularizers import l2

from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.metrics import classification_report, confusion_matrix
!pip install -q efficientnet

import efficientnet.tfkeras as efn
num_classes = 11

input_shape = (224,224,3)
# Build Model

net = efn.EfficientNetB7(input_shape=input_shape, weights='imagenet', include_top=False)



# add two FC layers (with L2 regularization)

x = net.output

x = GlobalAveragePooling2D()(x)



x = Dense(256)(x)

x = Dense(32)(x)



# Output layer

out = Dense(num_classes, activation="softmax")(x)



model = Model(inputs=net.input, outputs=out)

model.summary()
# Compile Model

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
## set Checkpoint : save best only, verbose on

#checkpoint = ModelCheckpoint("food11_efficientnetB7.hdf5", monitor='accuracy', verbose=0, save_best_only=True, mode='auto', save_freq=1)
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size

STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

STEP_SIZE_TEST =test_generator.n//test_generator.batch_size

num_epochs= 20
# Train Model

history = model.fit_generator(train_generator,steps_per_epoch=STEP_SIZE_TRAIN,epochs=num_epochs, validation_data=valid_generator, validation_steps=STEP_SIZE_VALID) #, callbacks=[checkpoint])
## Save Model

model.save('food11_efficientnetB7.h5')
score = model.evaluate(test_generator)
predY=model.predict(test_generator)

y_pred = np.argmax(predY,axis=1)

#y_label= [labels[k] for k in y_pred]

y_actual = test_generator.classes

cm = confusion_matrix(y_actual, y_pred)

print(cm)
print(classification_report(y_actual, y_pred, target_names=labels))