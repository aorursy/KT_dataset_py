# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import os

print(os.listdir("/kaggle/input/intel-image-classification"))
import cv2

from PIL import Image
basedir = os.listdir('/kaggle/input/intel-image-classification')
basedir
folder = basedir[0]

type(folder)
CLASS_NAMES = os.listdir('/kaggle/input/intel-image-classification/seg_train/seg_train') 
from keras.models import Sequential 

from keras.layers import Flatten, Activation, Dense, Lambda,Dropout

from keras.layers.convolutional import Conv2D

from keras.layers.pooling import MaxPool2D

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

from keras.layers import Conv2D, MaxPooling2D
datagen = ImageDataGenerator(rescale=1./255)
# train_datagen = ImageDataGenerator(rescale = 1./255.,horizontal_flip=True,shear_range=0.2,  zoom_range=0.2, validation_split=0.1)



# train_generator=train_datagen.flow_from_directory(

#       '/tmp/train/seg_train/',

#       target_size=(150,150),

#       batch_size=64,

#       class_mode='sparse',

#       seed=2209,

#       subset='training'

    

# )
train_dir      = '/kaggle/input/intel-image-classification/seg_train/seg_train'

test_dir       = '/kaggle/input/intel-image-classification/seg_test/seg_test'

validation_dir = '/kaggle/input/intel-image-classification/seg_pred'
train_generator = datagen.flow_from_directory(train_dir,

                                             target_size= (50,50),

                                             batch_size=20,

                                             shuffle=True,

                                             class_mode='categorical')
validation_generator = datagen.flow_from_directory(test_dir,

                                             target_size= (50,50),

                                             batch_size=20,

                                             class_mode='categorical')
augmented_images = [train_generator[i][0][0] for i in range(5)]
import matplotlib.pyplot as plt

def plotImages(images_arr):

    fig, axes = plt.subplots(1, 5, figsize=(20,20))

    axes = axes.flatten()

    for img, ax in zip( images_arr, axes):

        ax.imshow(img)

        ax.axis('off')

    plt.tight_layout()

    plt.show()
plotImages(augmented_images)
code = {"0":'buildings' ,"1":'forest',"2": 'glacier',"3": 'mountain',"4":'sea',"5":'street'}
model=Sequential()

model.add(Conv2D(filters=16,kernel_size=3,padding="same",activation="relu",input_shape=(50,50,3)))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32,kernel_size=3,padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64,kernel_size=3,padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64,kernel_size=3,padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())

model.add(Dense(6,activation="softmax"))

model.summary()

batch_size = train_generator.batch_size
model.compile(optimizer='adam',

             loss='categorical_crossentropy',

             metrics=['accuracy'])
train_sample = train_generator.samples

validation_sample = validation_generator.samples

# test_sample = test_generator.samples

steps_per_epoch = int(train_sample//batch_size)

validation_steps = int(validation_sample//batch_size)
history = model.fit_generator(train_generator,

                    steps_per_epoch = steps_per_epoch,

                    epochs = 10,

                    validation_data = validation_generator,

                    validation_steps = validation_steps

                             )
img = load_img('/kaggle/input/intel-image-classification/seg_train/seg_train/street/17374.jpg',target_size=(50,50))
plt.imshow(img)
img = np.array(img)/255

img = img.reshape(1,50,50,3)

pred = model.predict(img)

np.argmax(pred)

p = str(np.argmax(pred))

code[p]
load_img('/kaggle/input/intel-image-classification/seg_pred/seg_pred/9001.jpg',target_size=(128,128))
img = load_img('/kaggle/input/intel-image-classification/seg_pred/seg_pred/9001.jpg',target_size=(50,50))

# plt.imshow(img)

img = np.array(img)/255

img = img.reshape(1,50,50,3)

pred = model.predict(img)

np.argmax(pred)

p = str(np.argmax(pred))

code[p]
# list all data in history

print(history.history.keys())

# summarize history for accuracy

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
# needs  improvement this is a demo example 