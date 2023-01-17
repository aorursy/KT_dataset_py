import sys

import numpy as np

import cv2

import sklearn.metrics as sklm



from keras.applications.vgg16 import VGG16

from keras.preprocessing import image

from keras.applications.vgg16 import preprocess_input

from keras.layers import Input, Flatten, Dense

from keras.models import Model

from keras import optimizers

from keras.callbacks.callbacks import ModelCheckpoint



# from keras import backend as K

# img_dim_ordering = 'tf'

# K.set_image_dim_ordering(img_dim_ordering)

import os

base_dir = '/kaggle/input/duth-cv-2019-2020-hw-4/vehicles'

train_dir = os.path.join(base_dir, 'train')

validation_dir = os.path.join(base_dir, 'val')



# the model

def pretrained_model(img_shape, num_classes):

    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)

    #model_vgg16_conv.summary()

    

    #Create your own input format

    keras_input = Input(shape=img_shape, name = 'image_input')

    

    #Use the generated model 

    output_vgg16_conv = model_vgg16_conv(keras_input)

    

    #Add the fully-connected layers 

    x = Flatten(name='flatten')(output_vgg16_conv)

    x = Dense(128, activation='relu', name='fc2')(x)

    x = Dense(50, activation='relu', name='fc3')(x)

    x = Dense(num_classes, activation='softmax', name='predictions')(x)

    

    #Create your own model 

    pretrained_model = Model(inputs=keras_input, outputs=x)

    pretrained_model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['accuracy'])

    

    return pretrained_model



# loading the data

from tensorflow.keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(rescale=1./255)

val_datagen  = ImageDataGenerator(rescale=1./255)

flip_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)

rot_datagen = ImageDataGenerator(rescale=1./255,rotation_range=90)

zoom_datagen = ImageDataGenerator(rescale=1./255,zoom_range=0.5)

brightness_datagen = ImageDataGenerator(rescale=1./255,brightness_range=(0.0,1.0))





# --------------------

# Flow training images in batches of 20 using train_datagen generator

# --------------------

train_generator = train_datagen.flow_from_directory(train_dir,

                                                    batch_size=30,

                                                    class_mode='categorical',

                                                    # color_mode='grayscale',

                                                    target_size=(128,128),

                                                    shuffle=True)     

# --------------------

# Flow validation images in batches of 20 using test_datagen generator

# --------------------

validation_generator =  val_datagen.flow_from_directory(validation_dir,

                                                        batch_size=30,

                                                        class_mode='categorical',

                                                        #  color_mode='grayscale',

                                                         target_size=(128,128))



flip_generator = flip_datagen.flow_from_directory(train_dir,

                                                    batch_size=30,

                                                    class_mode='categorical',

                                                    # color_mode='grayscale',

                                                    target_size=(128,128),

                                                    shuffle=True)     

rot_generator = rot_datagen.flow_from_directory(train_dir,

                                                    batch_size=30,

                                                    class_mode='categorical',

                                                    # color_mode='grayscale',

                                                    target_size=(128,128),

                                                    shuffle=True)    

zoom_generator = zoom_datagen.flow_from_directory(train_dir,

                                                    batch_size=30,

                                                    class_mode='categorical',

                                                    # color_mode='grayscale',

                                                    target_size=(128,128),

                                                    shuffle=True)    

brightness_generator = brightness_datagen.flow_from_directory(train_dir,

                                                    batch_size=30,

                                                    class_mode='categorical',

                                                    # color_mode='grayscale',

                                                    target_size=(128,128),

                                                    shuffle=True)    



model = pretrained_model((128, 128, 3), 6)

model.compile(loss='categorical_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-4),

              metrics=['acc'])

#Train the model

checkpointer = ModelCheckpoint(filepath='small_last4.h5', verbose=1, save_best_only=True,monitor='val_acc')

history = model.fit_generator(

      train_generator,

      steps_per_epoch=train_generator.samples/train_generator.batch_size ,

      epochs=10,

      validation_data=validation_generator,

      validation_steps=validation_generator.samples/validation_generator.batch_size,

      verbose=1,callbacks=[checkpointer])

history2 = model.fit_generator(

      flip_generator,

      steps_per_epoch=flip_generator.samples/flip_generator.batch_size ,

      epochs=10,

      validation_data=validation_generator,

      validation_steps=validation_generator.samples/validation_generator.batch_size,

      verbose=1,callbacks=[checkpointer])

history3 = model.fit_generator(

      rot_generator,

      steps_per_epoch=rot_generator.samples/rot_generator.batch_size ,

      epochs=10,

      validation_data=validation_generator,

      validation_steps=validation_generator.samples/validation_generator.batch_size,

      verbose=1,callbacks=[checkpointer])

history4 = model.fit_generator(

      zoom_generator,

      steps_per_epoch=zoom_generator.samples/zoom_generator.batch_size ,

      epochs=10,

      validation_data=validation_generator,

      validation_steps=validation_generator.samples/validation_generator.batch_size,

      verbose=1,callbacks=[checkpointer])

history5 = model.fit_generator(

      brightness_generator,

      steps_per_epoch=brightness_generator.samples/brightness_generator.batch_size,

      epochs=10,

      validation_data=validation_generator,

      validation_steps=validation_generator.samples/validation_generator.batch_size,

      verbose=1,callbacks=[checkpointer])



# Save the model

model.save('small_last4.h5')
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import tensorflow as tf

from keras.preprocessing import image

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import csv



model = tf.keras.models.load_model(filepath='small_last4.h5')

rowlist = [['Id', 'Category']]



for dirname, _, filenames in os.walk('/kaggle/input/duth-cv-2019-2020-hw-4/vehicles/test'):

    for filename in filenames:

        path = os.path.join(dirname, filename)

        img = image.load_img(path, target_size=(128, 128), grayscale=False, interpolation='bilinear')

        

        x = image.img_to_array(img)

        x = np.expand_dims(x, axis=0)

        

        classes_pred = model.predict(x)

        cls_pred = np.argmax(classes_pred)

        rowlist.append([filename, cls_pred])

        print(filename, cls_pred)

        with open('output.csv', 'w', newline='') as file:

            writer = csv.writer(file)

            writer.writerows(rowlist)