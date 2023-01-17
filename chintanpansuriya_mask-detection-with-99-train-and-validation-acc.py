import os

import random

import tensorflow as tf

from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.preprocessing.image import ImageDataGenerator
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')

])



model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])

print (os.listdir('/kaggle/input/facemask-detection-dataset-20000-images/'))
TRAINING_DIR = "/kaggle/input/facemask-detection-dataset-20000-images/"

train_datagen = ImageDataGenerator(rescale=1.0/255.,validation_split=0.25,  

                                  horizontal_flip=True,

                                  vertical_flip=True,

                                  featurewise_center=True,

                                  rotation_range=20,

                                  width_shift_range=0.15,

                                  height_shift_range=0.25,

                                  shear_range=0.2,

                                  zoom_range=0.2,

                                  )



train_generator = train_datagen.flow_from_directory(TRAINING_DIR,

                                                    class_mode='binary',

                                                    subset = 'training',

                                                    target_size=(150, 150))



validation_generator = train_datagen.flow_from_directory(TRAINING_DIR,

                                                         class_mode='binary',

                                                         subset = 'validation',

                                                         target_size=(150, 150))
history = model.fit(train_generator, epochs=5, validation_data=validation_generator)
%matplotlib inline



import matplotlib.pyplot as plt



acc=history.history['acc']

val_acc=history.history['val_acc']

loss=history.history['loss']

val_loss=history.history['val_loss']



epochs=range(len(acc))



plt.plot(epochs, acc, 'r', "Training Accuracy")

plt.plot(epochs, val_acc, 'b', "Validation Accuracy")

plt.title('Training and validation accuracy')

plt.figure()



plt.plot(epochs, loss, 'r', "Training Loss")

plt.plot(epochs, val_loss, 'b', "Validation Loss")

plt.figure()