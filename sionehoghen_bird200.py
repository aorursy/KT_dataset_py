import os

valid = len(os.listdir('/kaggle/input/100-bird-species/valid'))

consolidated = len(os.listdir('/kaggle/input/100-bird-species/consolidated'))

test = len(os.listdir('/kaggle/input/100-bird-species/test'))

train = len(os.listdir('/kaggle/input/100-bird-species/train'))
print(valid, consolidated, test, train)
valid_path = '/kaggle/input/100-bird-species/valid/'

consolidated_path = '/kaggle/input/100-bird-species/consolidated/'

test_path = '/kaggle/input/100-bird-species/test/'

train_path = '/kaggle/input/100-bird-species/train/'

total_no = [valid, consolidated, test, train]

valid = (os.listdir('/kaggle/input/100-bird-species/valid/'))

consolidated = (os.listdir('/kaggle/input/100-bird-species/consolidated/'))

test = (os.listdir('/kaggle/input/100-bird-species/test/'))

train = (os.listdir('/kaggle/input/100-bird-species/train/'))

total_no = [valid, consolidated, test, train]

path_list = [valid_path, consolidated_path, test_path, train_path]
# Checking the no of images in every sets.

i = 0

valid_no, consolidated_no, test_no, train_no = 0, 0, 0, 0

y = [valid_no, consolidated_no, test_no, train_no]

j = 0

for file in total_no:

    for f in file:

        x = (len(os.listdir(path_list[i] + f)))

        y[j] = x + y[j]

        

    i +=1

    j +=1

print(y)
import pandas as pd

import numpy as np

import matplotlib.pyplot as mat

import matplotlib.image as img

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.preprocessing.image import load_img 

import seaborn as sns
load_img('/kaggle/input/100-bird-species/train/ALBATROSS/009.jpg')
height, width, channels, filename  =[], [], [], []

for f in train:

    x = (os.listdir(train_path + f))

    y = train_path + f

    for image in x:

        shapee = img.imread(y +'/'+ image).shape

        filename.append(image)

        height.append(shapee[0])

        width.append(shapee[1])

        channels.append(shapee[2])

        
df_train = pd.DataFrame({'Filename': filename, 'Height': height, 'Width': width, 'Channels': channels}) 
df_train.describe()
train_ds = keras.preprocessing.image_dataset_from_directory(train_path, image_size = (180,180))

test_ds = keras.preprocessing.image_dataset_from_directory(test_path, image_size = (180,180))

valid_ds = keras.preprocessing.image_dataset_from_directory(valid_path, image_size = (180, 180))

cosolidated_ds = keras.preprocessing.image_dataset_from_directory(consolidated_path, image_size = (180, 180))
mat.figure(figsize=(10,10))

for image, labels in train_ds.take(1):

    for i in range(9):

        mat.subplot(3, 3, i+1)

        mat.imshow(image[i].numpy().astype('uint8'))

        mat.axis(False)

mat.show()
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rotation_range= 30, rescale = 1/255.)

test_datagen = ImageDataGenerator(rescale = 1/255.)

valid_datagen = ImageDataGenerator(rescale = 1/255.)

consolidated_datagen = ImageDataGenerator(rescale = 1/255, rotation_range = 20, horizontal_flip= True)



train_data = train_datagen.flow_from_directory(train_path, target_size = (180,180))

test_data = test_datagen.flow_from_directory(test_path, target_size = (180, 180))

valid_data = valid_datagen.flow_from_directory(valid_path, target_size = (180, 180))

consolidated_data = consolidated_datagen.flow_from_directory(consolidated_path, target_size = (180, 180))
mat.figure(figsize = (10,10))

for image, labels in train_data:

    for i in range(9):

        mat.subplot(3,3, i+1)

        mat.imshow(image[i])

    mat.show()

    break
# Build model



from keras.layers import Conv2D, Dense, Flatten, BatchNormalization, Dropout, Activation, MaxPooling2D



model = keras.models.Sequential()

model.add(Conv2D(32, kernel_size = 7,input_shape = (180,180,3), padding = 'same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size = 2))

model.add(Dropout(0.3))



model.add(Conv2D(64, kernel_size = 3, padding = 'same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size = 2))

model.add(Dropout(0.3))



model.add(Conv2D(128, kernel_size = 3, padding = 'same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size = 2))

model.add(Dropout(0.3))



model.add(Conv2D(256, kernel_size = 3, padding = 'same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size = 2))

model.add(Dropout(0.3))



model.add(Conv2D(512, kernel_size = 3, padding = 'same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size = 2))

model.add(Dropout(0.3))



model.add(Flatten())



model.add(Dense(256))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dropout(0.3))



model.add(Dense(128))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dropout(0.3))



model.add (Dense(225, activation = 'softmax'))
model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
checkpoints = keras.callbacks.ModelCheckpoint('/kaggle/input/Model_final.h5', save_best_only= True, save_weights_only=True, verbose = 1, monitor='val_accuracy')

learning_rate = keras.callbacks.ReduceLROnPlateau(patience = 3, monitor = 'val_loss', factor = 0.1, min_lr=0.00001)

from livelossplot import PlotLossesKeras

tensor = PlotLossesKeras()

callbacks = [checkpoints, learning_rate, tensor]
model.fit(train_data, epochs = 20, validation_data= valid_data, callbacks = callbacks)