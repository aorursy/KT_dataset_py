# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Imported libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.utils import shuffle

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split

import keras

import cv2

from imageio import imread

from PIL import Image

from keras import layers

from keras import models

from keras.models import Sequential, Model, load_model

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D

from keras.layers import Conv2D, MaxPooling2D

from keras.layers.normalization import BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from keras.applications import VGG16, ResNet50







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

from os.path import join

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
img_dir = "../input/flowers/flowers/"
img_dir
flower_list=os.listdir(img_dir)

flower_list
sunflower_dir='../input/flowers/flowers/sunflower'

tulip_dir='../input/flowers/flowers/tulip'

daisy_dir='../input/flowers/flowers/daisy'

rose_dir='../input/flowers/flowers/rose'

dandelion_dir='../input/flowers/flowers/dandelion'
data = []

labels = []



for u in os.listdir(sunflower_dir):

    try:

        

        image = imread("../input/flowers/flowers/sunflower"+"/"+u)

        image_array = Image.fromarray(image, 'RGB')

        resize_img = image_array.resize((224 , 224))

        data.append(np.array(resize_img))

        labels.append(0)

        

    except AttributeError:

        print('')

        

for v in os.listdir(tulip_dir)[:800]:

    try:

        

        image = imread("../input/flowers/flowers/tulip"+"/"+v)

        image_array = Image.fromarray(image , 'RGB')

        resize_img = image_array.resize((224 , 224))

        data.append(np.array(resize_img))

        labels.append(1)

        

    except AttributeError:

        print('')



for j in os.listdir(daisy_dir):

    try:

        

        image = imread("../input/flowers/flowers/daisy"+"/"+j)

        image_array = Image.fromarray(image , 'RGB')

        resize_img = image_array.resize((224 , 224))

        data.append(np.array(resize_img))

        labels.append(2)

        

    except AttributeError:

        print('')



for k in os.listdir(rose_dir):

    try:

        

        image = imread("../input/flowers/flowers/rose"+"/"+k)

        image_array = Image.fromarray(np.array(image) , 'RGB')

        resize_img = image_array.resize((224 , 224))

        data.append(np.array(resize_img))

        labels.append(3)

        

    except AttributeError:

        print('') 



for l in os.listdir(dandelion_dir)[:800]:

    if l.endswith('.jpg'):

        try:

        

            image = imread("../input/flowers/flowers/dandelion"+"/"+l)

            image_array = Image.fromarray(image , 'RGB')

            resize_img = image_array.resize((224 , 224))

            data.append(np.array(resize_img))

            labels.append(4)

        

        except AttributeError:

            print('') 
flowers = np.array(data)

labels = np.array(labels)
np.shape(flowers)
print('flowers : {} | labels : {}'.format(flowers.shape , labels.shape))
def flower_type(x):

    if x==0:

        return 'sunflower'

    if x==1: 

        return 'tulip'

    if x==2:

        return 'daisy'

    if x==3:

        return 'rose'

    else:

        return 'dandelion'





plt.figure(1 , figsize = (15 , 9))

n = 0 

for i in range(25):

    n += 1 

    r = np.random.randint(0 , flowers.shape[0] , 1)

    plt.subplot(5 , 5 , n)

    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)

    plt.imshow(flowers[r[0]])

    plt.title('{} : {}'.format(flower_type(labels[r[0]]), labels[r[0]]))

    plt.xticks([]) , plt.yticks([])

    

plt.show()

flowers, labels=shuffle(flowers, labels, random_state=13)
X_train, X_test, y_train, y_test = train_test_split(flowers, labels, test_size=0.2, random_state=13)



X_train, X_val, y_train, y_val= train_test_split(X_train, y_train, test_size=0.2, random_state=13)
#import inception with pre-trained weights. do not include fully #connected layers

inception_base = ResNet50(weights='imagenet', include_top=False)



# add a global spatial average pooling layer

x = inception_base.output

x = GlobalAveragePooling2D()(x)

x = Dropout(0.5)(x)

# add a fully-connected layer

x = Dense(512, activation='relu')(x)

# and a fully connected output/classification layer

predictions = Dense(5, activation='softmax')(x)

# create the full network so we can train on it

inception_transfer = Model(inputs=inception_base.input, outputs=predictions)
#import inception with pre-trained weights. do not include fully #connected layers

inception_base_vanilla = ResNet50(weights=None, include_top=False)



# add a global spatial average pooling layer

x = inception_base_vanilla.output

x = GlobalAveragePooling2D()(x)

# add a fully-connected layer

x = Dense(512, activation='relu')(x)

# and a fully connected output/classification layer

predictions = Dense(5, activation='softmax')(x)

# create the full network so we can train on it

inception_transfer_vanilla = Model(inputs=inception_base_vanilla.input, outputs=predictions)
inception_transfer.compile(loss='categorical_crossentropy',

              optimizer=keras.optimizers.RMSprop(lr=2e-5),

              metrics=['accuracy'])



inception_transfer_vanilla.compile(loss='categorical_crossentropy',

              optimizer=keras.optimizers.RMSprop(lr=2e-5),

              metrics=['accuracy'])
inception_transfer.summary()
batch_size=16



num_classes=5

y_train = to_categorical(y_train, num_classes)

y_val = to_categorical(y_val, num_classes)

y_test = to_categorical(y_test, num_classes)





train_datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    fill_mode='nearest')



test_datagen = ImageDataGenerator(rescale=1./255)



train_datagen.fit(X_train)



train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)



validation_generator=test_datagen.flow(X_val, y_val, batch_size=batch_size)



history = inception_transfer.fit_generator(

      train_generator,    

      steps_per_epoch=len(X_train) / batch_size, 

      epochs=30,

      validation_data=validation_generator,

      validation_steps=50)

                    
import matplotlib.pyplot as plt

acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()