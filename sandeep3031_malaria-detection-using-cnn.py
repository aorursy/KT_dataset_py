import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import os

# Importing Keras library

from PIL import Image

import cv2



#importing the keres libraries

from keras.models import Model, Sequential

from keras.layers import Flatten, Dense, Dropout

from keras.layers import Convolution2D, MaxPooling2D

from keras.layers import BatchNormalization, GlobalAveragePooling2D

from keras.utils import to_categorical

from keras.optimizers import Adam



import warnings

warnings.filterwarnings("ignore")
DATA_DIR="../input/cell-images-for-detecting-malaria/cell_images/"

SIZE=64

dataset=[]

label=[]

parasitized_images=os.listdir(DATA_DIR + 'Parasitized/')

for i,image_name in enumerate(parasitized_images):

    try:

        if (image_name.split('.')[1]=="png"):

            image=cv2.imread(DATA_DIR + 'Parasitized/' + image_name)

            image=Image.fromarray(image,"RGB")

            image=image.resize((64,64))

            dataset.append(np.array(image))

            label.append(0)

    except Exception:

        None

        
uninfected_images=os.listdir(DATA_DIR + 'Uninfected/')

for i,image_name in enumerate(uninfected_images):

    try:

        if (image_name.split('.')[1]=="png"):

            image=cv2.imread(DATA_DIR + 'Uninfected/' + image_name)

            image=Image.fromarray(image,"RGB")

            image=image.resize((64,64))

            dataset.append(np.array(image))

            label.append(1)

    except Exception:

        None
#visualizing the Parasitized images. 

plt.figure(figsize=(18,12))

for index,image_index in enumerate(np.random.randint(len(parasitized_images),size=5)):

    plt.subplot(1,5,index+1)

    plt.imshow(dataset[image_index])
#visualizing the uninfected images.

plt.figure(figsize=(18,12))

for index,image_index in enumerate(np.random.randint(len(uninfected_images),size=5)):

    plt.subplot(1,5,index+1)

    plt.imshow(dataset[len(parasitized_images)+image_index])
#splitting the data into train and test

from sklearn.model_selection import train_test_split

train_X,test_X,train_y,test_y=train_test_split(dataset,to_categorical(np.array(label)),test_size=0.30,random_state=123)
from keras.preprocessing.image import ImageDataGenerator

train_generator = ImageDataGenerator(rescale = 1/255,

                                     zoom_range = 0.3,

                                     horizontal_flip = True,

                                     rotation_range = 30)



test_generator = ImageDataGenerator(rescale = 1/255)



train_generator = train_generator.flow(np.array(train_X),

                                       train_y,

                                       batch_size = 64,

                                       shuffle = False)



test_generator = test_generator.flow(np.array(test_X),

                                     test_y,

                                     batch_size = 64,

                                     shuffle = False)
model = Sequential()

model.add(Convolution2D(32, (3, 3), input_shape = (SIZE, SIZE, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2, 2), data_format="channels_last"))

model.add(BatchNormalization(axis = -1))

model.add(Dropout(0.2))



model.add(Convolution2D(32, (3, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2, 2), data_format="channels_last"))

model.add(BatchNormalization(axis = -1))

model.add(Dropout(0.2))

model.add(Flatten())



model.add(Dense(activation = 'relu', units=512))

model.add(BatchNormalization(axis = -1))

model.add(Dropout(0.2))



model.add(Dense(activation = 'relu', units=256))

model.add(BatchNormalization(axis = -1))

model.add(Dropout(0.2))



model.add(Dense(activation = 'sigmoid', units=2))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
history=model.fit(train_generator,validation_data=test_generator,epochs=10,verbose=1,shuffle=False)
#Plotiing the Loss and Accuracy of our Model.

train_acc = history.history['accuracy']

train_loss = history.history['loss']



val_acc = history.history['val_accuracy']

val_loss = history.history['val_loss']



plt.figure(figsize=(20,5))

plt.subplot(1,2,1)

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.plot(train_loss)

plt.plot(val_loss)



plt.subplot(1,2,2)

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.plot(train_acc)

plt.plot(val_acc)