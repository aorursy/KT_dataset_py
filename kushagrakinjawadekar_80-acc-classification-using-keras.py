# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/dogs-cats-images'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
TRAINING_DIR = '/kaggle/input/dogs-cats-images/dog vs cat/dataset/training_set'

VALIDATION_DIR = '/kaggle/input/dogs-cats-images/dog vs cat/dataset/test_set'
from tensorflow.keras.preprocessing import image

import matplotlib.pyplot as plt

img = image.load_img('/kaggle/input/dogs-cats-images/dataset/training_set/dogs/dog.1789.jpg')

plt.imshow(img)
from keras.preprocessing.image import ImageDataGenerator

width = 256

height = 256

batch_size = 100 



train_datagen = ImageDataGenerator(

                rescale = 1./255.0,                            

                shear_range=0,

                zoom_range=0,

                horizontal_flip=False,

                width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

                height_shift_range=0.1)  # randomly shift images vertically (fraction of total height)





train_generator = train_datagen.flow_from_directory(

                    TRAINING_DIR,

                    target_size=(width, height),

                    batch_size=batch_size,

                    class_mode = 'binary',

                    shuffle=True)



vaid_datagen = ImageDataGenerator(

                rescale=1.0/255.0)



valid_generator = vaid_datagen.flow_from_directory(

                  VALIDATION_DIR,

                  target_size=(width, height),

                  class_mode = 'binary',

                  batch_size=batch_size  )
import tensorflow as tf

model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape = (256,256,3)),

    tf.keras.layers.MaxPooling2D(pool_size = (2,2)),

    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),

    tf.keras.layers.MaxPooling2D(pool_size = (2,2)),

    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),

    tf.keras.layers.MaxPooling2D(pool_size = (2,2)),

    

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512,activation='relu'),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(256,activation='relu'),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Dense(1,activation='sigmoid')

    

    

])
class myCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):

        if(logs.get('accuracy')>0.9600):

            print("\nReached 96% accuracy so cancelling training!")

            self.model.stop_training = True
model.compile(loss='binary_crossentropy',

              optimizer='rmsprop',

              metrics=['accuracy'])
callbacks = myCallback()

history = model.fit_generator(

        train_generator,

        steps_per_epoch=8000 // batch_size,

        epochs=25,

        validation_data=valid_generator,

        validation_steps=2000 // batch_size,

        callbacks=[callbacks]

        )
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1, len(accuracy) + 1)

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