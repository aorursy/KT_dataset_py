import os

import keras

import random

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import tensorflow as tf

from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import Dropout


print(os.listdir('/kaggle/input/intel-image-classification'))
TRAINING_DIR = '/kaggle/input/intel-image-classification/seg_train/seg_train'
train_datagen = ImageDataGenerator(

      rescale=1./255,

      rotation_range=40,

      width_shift_range=0.2,

      height_shift_range=0.2,

      shear_range=0.2,

      zoom_range=0.2,

      horizontal_flip=True,

      fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(TRAINING_DIR,

                                                    batch_size=32,

                                                    class_mode='categorical',

                                                    target_size=(150, 150))
print(train_generator.class_indices)
VALIDATION_DIR = '/kaggle/input/intel-image-classification/seg_test/seg_test'

validation_datagen = ImageDataGenerator(rescale=1./255.)

validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,

                                                              batch_size=32,

                                                              class_mode='categorical',

                                                              target_size=(150, 150))
print(validation_generator.class_indices)
model = tf.keras.models.Sequential([

                tf.keras.layers.Conv2D(96, (11,11), activation='relu', input_shape=(150, 150, 3), padding='same'),

                tf.keras.layers.MaxPooling2D(3,3),

                Dropout(0.2),

                tf.keras.layers.Conv2D(64, (7,7), activation='relu'),

                tf.keras.layers.MaxPooling2D(3,3),

                Dropout(0.2),

                tf.keras.layers.Conv2D(32, (5,5), activation='relu'),

                Dropout(0.2),

                tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

                tf.keras.layers.MaxPooling2D(2,2),

                tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

                tf.keras.layers.MaxPooling2D(2,2),               

                tf.keras.layers.Flatten(),

                tf.keras.layers.Dense(512, activation='relu'),

                Dropout(0.2),

                tf.keras.layers.Dense(256, activation='relu'),

                Dropout(0.2),

                tf.keras.layers.Dense(6, activation='softmax')

                ])

model.summary()
model.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['acc'])
history = model.fit_generator(train_generator,

                              epochs=50,

                              verbose=1,

                              steps_per_epoch=len(train_generator)//16,

                              validation_data=validation_generator)

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()

test_image=random.choice(os.listdir("/kaggle/input/intel-image-classification/seg_pred/seg_pred/"))

test_image=os.path.join("/kaggle/input/intel-image-classification/seg_pred/seg_pred/", test_image)
image = tf.keras.preprocessing.image.load_img(test_image)

input_arr = keras.preprocessing.image.img_to_array(image)

input_arr = np.array([input_arr])  

predictions = model.predict_classes(input_arr)
predictions
import matplotlib.pyplot as plt

import matplotlib.image as mpimg

img=mpimg.imread(test_image)

plt.axis('off')

imgplot = plt.imshow(img)

plt.show()

if predictions==0:

    print('it is a building')

elif predictions==1:

    print('it is forest')

elif predictions==2:

    print('it is a glacier')

elif predictions==3:

    print('it is a mountain')

elif predictions==4:

    print('it is sea')

elif predictions==5:

    print('it is a street')