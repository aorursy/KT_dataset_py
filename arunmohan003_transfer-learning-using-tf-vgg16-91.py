import tensorflow as tf

import tensorflow.keras as keras

import numpy as np

import os

import matplotlib.pyplot as plt

import pandas as pd

from tensorflow.keras.applications import VGG16

from tensorflow.keras.optimizers import Adam

from tensorflow.keras import layers

from tensorflow.keras import Model

from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray/train'

val_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray/val'

test_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray/test'

# len(os.listdir('/kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA'))
ls /kaggle/input/chest-xray-pneumonia/chest_xray/test
nb_train_samples = len(os.listdir(train_dir + '/NORMAL')) + len(os.listdir(train_dir + '/PNEUMONIA'))

print(nb_train_samples)

nb_val_samples  = len(os.listdir(val_dir + '/NORMAL')) + len(os.listdir(val_dir + '/PNEUMONIA'))

print(nb_val_samples)
train_datagen = ImageDataGenerator(rescale = 1./255,

                                      shear_range=0.2,zoom_range=0.2,

                                      horizontal_flip=True)



val_datagen = ImageDataGenerator(rescale = 1./255)



test_datagen = ImageDataGenerator(rescale=1./ 255)



batch_size = 10

img_width = 150

img_height = 150



train_generator = train_datagen.flow_from_directory(train_dir,batch_size = batch_size,

                                                    class_mode = 'binary', 

                                                    target_size =(img_width, img_height))



val_generator = val_datagen.flow_from_directory(val_dir,batch_size = batch_size,

                                                    class_mode = 'binary', 

                                                    target_size =(img_width, img_height))







test_generator = test_datagen.flow_from_directory(test_dir,batch_size=batch_size,                                              

                                                class_mode='binary',

                                                target_size=(img_width, img_height))



pre_trained_model = VGG16(input_shape = (150, 150, 3), 

                                include_top = False, 

                                weights = 'imagenet')



pre_trained_model.summary()
for layer in pre_trained_model.layers:

      layer.trainable = False


last_layer = pre_trained_model.get_layer('block4_pool')



print('Shape of last layer {}'.format(last_layer.output_shape))

last_output = last_layer.output




# Flatten the output layer to 1 dimension

x = layers.Flatten()(last_output)

# Add a fully connected layer with 1,024 hidden units and ReLU activation

x = layers.Dense(1024, activation='relu')(x)

# Add a dropout rate of 0.2

x = layers.Dropout(0.2)(x)                  

# Add a final sigmoid layer for classification

x = layers.Dense(1, activation='sigmoid')(x)           



model = Model(pre_trained_model.input, x) 



model.summary()
model.compile(optimizer = Adam(lr=0.0001), 

              loss = 'binary_crossentropy', 

              metrics = ['acc'])
history = model.fit_generator(

            train_generator,

            validation_data = val_generator,

            steps_per_epoch = nb_train_samples // batch_size,

            epochs = 5,

            validation_steps = nb_val_samples // batch_size,

            verbose = 1)
import matplotlib.pyplot as plt

acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend(loc=0)

plt.figure()





plt.show()
# evaluate the model

scores = model.evaluate_generator(test_generator)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))