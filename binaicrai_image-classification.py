# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



import warnings

warnings.simplefilter('ignore')

warnings.filterwarnings('ignore')



import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

%matplotlib inline



import shutil, sys

import tensorflow as tf

from keras.utils import to_categorical



from PIL import Image



from keras.models import Model, Sequential

from keras.layers import Flatten, Dense, Dropout, Activation

from keras.layers import Convolution2D, MaxPooling2D, Conv2D

from keras.layers import BatchNormalization, GlobalAveragePooling2D

from keras.utils import to_categorical

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

from keras.utils.data_utils import Sequence

from keras.layers.convolutional import Conv2D



# Get Inception architecture from keras.applications

import scipy

from keras.applications.inception_v3 import InceptionV3

from scipy.misc import *
path = '../input/rapid-sketch/Rapid Sketch/'   # Train data path

#val_path = 'data/validation'       # Validation data path

classes = os.listdir(path)

#del classes[0]

# List of directories in train path

print(classes)
class_map = {'Props' : 'Props', 'BentKnee': 'BentKnee', 'LyingDown': 'LyingDown', 'Standing': 'Standing'}
datagen = ImageDataGenerator(

        rotation_range=30,

        width_shift_range=0.2,

        height_shift_range=0.2,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True,

        fill_mode='nearest')
aug_images_path = 'preview'

f = os.listdir(os.path.join(path,classes[3]))[0]

img_path = os.path.join(path,classes[3],f)

img = load_img(img_path)       

x = img_to_array(img)          # this is a Numpy array 

print(x.shape)

x = x.reshape((1,) + x.shape)  # this is a Numpy array 

print(x.shape)



# Create a directory named 'preview' to save augmented images. 

# Delete, if already exists

if os.path.isdir(aug_images_path):

#     os.system('rm -rf '+aug_images_path)

    shutil.rmtree(aug_images_path)

    

os.system('mkdir '+aug_images_path)

    

# the .flow() command below generates augmented images and saves them to a directory names 'preview'

i = 0

for batch in datagen.flow(x, batch_size=1, save_to_dir=aug_images_path, save_prefix='c0', save_format='jpg'):

    i += 1

    if i > 9:

        break



plt.imshow(img)

plt.axis('off')

plt.title('Original Image: '+f)
# Plot the augmented images for the above original image

# Read them from 'preview' directory and display them



plt.figure(figsize=(20,6))

aug_images = os.listdir('preview')

for ix,i in enumerate(aug_images):

    img = mpimg.imread(os.path.join('preview',i))

    plt.subplot(2,5,ix+1)

    plt.imshow(img)

    plt.axis('off')

    plt.title(i)

    if ix==10:

        break
datagen = ImageDataGenerator(rescale=1./255,

                            validation_split = 0.2,rotation_range=30,

                                    width_shift_range=0.2,

                                    height_shift_range=0.2,

                                    shear_range=0.2,

                                    zoom_range=0.2,

                                    horizontal_flip=True,

                                    fill_mode='nearest')

# flow_from_directory gets label for an image from the sub-directory it is placed in

# Generate Train data

train_generator = datagen.flow_from_directory(

        path,

        target_size=(224, 224),

        batch_size=64,

        subset='training',

        class_mode='categorical')



# Generate Validation data

val_generator = datagen.flow_from_directory(

        path,

        target_size=(224, 224),

        batch_size=64,

        subset='validation',

        class_mode='categorical')
plt.figure(figsize=(20,6))

for ix,i in enumerate(classes):

    f = os.listdir(os.path.join(path,i))[0]

    img = mpimg.imread(os.path.join(path,i,f))

    plt.subplot(2,5,ix+1)

    plt.imshow(img)

    plt.axis('off')

    plt.title(class_map[i])
def image_classifier(nb_classes):

    model = Sequential()



    model.add(Conv2D(filters=32, kernel_size=(5, 5), input_shape=(224, 224, 3), padding='valid'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))



    model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='valid'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))



    model.add(Conv2D(filters=128, kernel_size=(5, 5), padding='valid'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Flatten())

    model.add(Dropout(0.2))



    model.add(Dense(128, init='uniform', activation='relu'))

    model.add(Dropout(0.4))



    model.add(Dense(nb_classes, activation='softmax'))

    

    return(model)
model = image_classifier(nb_classes=4)

model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
hist0 = model.fit_generator(train_generator, 

                           validation_data=val_generator, 

                           epochs=30,

                           steps_per_epoch=120/64.0,validation_steps=45/64.0,).history
plt.figure(figsize=(20,8))

plt.subplot(1,2,1)

train_loss = plt.plot(hist0['loss'], label='train loss')

val_loss = plt.plot(hist0['val_loss'], label='val loss')

plt.legend()

plt.title('Loss')



plt.subplot(1,2,2)

train_loss = plt.plot(hist0['accuracy'], label='train acc')

val_loss = plt.plot(hist0['val_accuracy'], label='val acc')

plt.legend()

plt.title('Accuracy')
val_preds = model.predict_generator(generator=val_generator, steps=45/64.0)

val_preds = np.array(val_preds).flatten()

val_preds_class = np.array(val_preds.argmax(axis=0)).flatten().tolist()

#val_preds_class = np.array(val_preds).flatten().tolist()

a = {'image':val_generator.filenames, 'prediction':val_preds_class}

val_preds_df = pd.DataFrame.from_dict(a, orient = 'index')

val_preds_df = val_preds_df.transpose()
val_preds_df.head(20)
val_generator.class_indices
datagen = ImageDataGenerator(rescale=1./255,

                            validation_split = 0.2,rotation_range=30,

                                    width_shift_range=0.2,

                                    height_shift_range=0.2,

                                    shear_range=0.2,

                                    zoom_range=0.2,

                                    horizontal_flip=True,

                                    fill_mode='nearest')

# flow_from_directory gets label for an image from the sub-directory it is placed in

# Generate Train data

train_generator_2 = datagen.flow_from_directory(

        path,

        target_size=(224, 224),

        batch_size=64,

        subset='training',

        class_mode='categorical')



# Generate Validation data

val_generator_2 = datagen.flow_from_directory(

        path,

        target_size=(224, 224),

        batch_size=64,

        subset='validation',

        class_mode='categorical')
# Get Inception architecture from keras.applications

from keras.applications.inception_v3 import InceptionV3



def inception_tl(nb_classes, freez_wts):

    

    trained_model = InceptionV3(include_top=False,weights='imagenet')

    x = trained_model.output

    x = GlobalAveragePooling2D()(x)

    pred_inception= Dense(nb_classes,activation='softmax')(x)

    model = Model(inputs=trained_model.input,outputs=pred_inception)

    

    for layer in trained_model.layers:

        layer.trainable=(1-freez_wts)

    

    return(model)
model2 = inception_tl(nb_classes=4, freez_wts=False)

model2.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
hist1 = model2.fit_generator(train_generator_2, 

                           validation_data=val_generator_2, 

                           epochs=15,

                           steps_per_epoch=120/64.0,validation_steps=45/64.0,).history
hist2 = model2.fit_generator(train_generator, 

                           validation_data=val_generator, 

                           epochs=10,

                           steps_per_epoch=120/64.0,validation_steps=45/64.0,).history
plt.figure(figsize=(20,8))

plt.subplot(1,2,1)

train_loss = plt.plot(hist1['loss'], label='train loss')

val_loss = plt.plot(hist1['val_loss'], label='val loss')

plt.legend()

plt.title('Loss')



plt.subplot(1,2,2)

train_loss = plt.plot(hist1['accuracy'], label='train acc')

val_loss = plt.plot(hist1['val_accuracy'], label='val acc')

plt.legend()

plt.title('Accuracy')
plt.figure(figsize=(20,8))

plt.subplot(1,2,1)

train_loss = plt.plot(hist2['loss'], label='train loss')

val_loss = plt.plot(hist2['val_loss'], label='val loss')

plt.legend()

plt.title('Loss')



plt.subplot(1,2,2)

train_loss = plt.plot(hist2['accuracy'], label='train acc')

val_loss = plt.plot(hist2['val_accuracy'], label='val acc')

plt.legend()

plt.title('Accuracy')
val_preds2 = model2.predict_generator(generator=val_generator_2, steps=45/64.0)

val_preds2 = np.array(val_preds2).flatten()

val_preds_class2 = np.array(val_preds2.argmax(axis=0)).flatten().tolist()

#val_preds_class = np.array(val_preds).flatten().tolist()

a = {'image':val_generator_2.filenames, 'prediction':val_preds_class2}

val_preds_df2 = pd.DataFrame.from_dict(a, orient = 'index')

val_preds_df2 = val_preds_df2.transpose()
val_preds_df2.head(20)
val_generator_2.class_indices