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

import tensorflow as tf

import keras_preprocessing

from tensorflow.keras.preprocessing import image

import pickle

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import TensorBoard

tf.__version__

# Any results you write to the current directory are saved as output.


class myCallback(tf.keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs={}):

    if(logs.get('val_accuracy')>0.95 and logs.get('val_loss')<0.2):

      print("\nReached 60% accuracy so cancelling training!")

      self.model.stop_training = True

callbacks = myCallback()
TRAINING_DIR = '/kaggle/input/tomato/New Plant Diseases Dataset(Augmented)/train/'

VALIDATION_DIR = '/kaggle/input/tomato/New Plant Diseases Dataset(Augmented)/valid/'



training_datagen = ImageDataGenerator(

													rescale = 1./255,

												rotation_range=40,

													width_shift_range=0.2,

													height_shift_range=0.2,

													shear_range=0.2,

													zoom_range=0.2,

													horizontal_flip=True,

													fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale = 1./255)





train_generator = training_datagen.flow_from_directory(

	TRAINING_DIR,

	target_size=(256,256),

	class_mode='categorical',

	color_mode="rgb",

	        batch_size=128,

	

)

validation_generator = validation_datagen.flow_from_directory(

	VALIDATION_DIR,

	target_size=(256,256),

	class_mode='categorical',

	color_mode="rgb"

)

#
model = tf.keras.models.Sequential([

    

    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(256, 256, 3)),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout( 0.25),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(512, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout( 0.5),

    tf.keras.layers.Conv2D(1024, (3,3), activation='relu'),

    tf.keras.layers.Dropout( 0.5),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(10, activation='softmax')

])
model.summary()
from datetime import datetime

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

tf.keras.callbacks.TensorBoard(

    log_dir='logs', histogram_freq=0, write_graph=True, write_images=False,

    update_freq='epoch', profile_batch=2, embeddings_freq=0,

    embeddings_metadata=None,)

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")



from time import time

model.compile(optimizer='adam',

              loss='categorical_crossentropy',

              metrics=['accuracy'])

tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs/{}".format(time()))

 
model.fit(train_generator, 

                          steps_per_epoch=len(train_generator),

                          validation_data=validation_generator,

                          epochs=50,verbose=1,

                callbacks=[tensorboard_callback,callbacks] )

# m odel.fit(x_train, y_train, epochs=5)


for j in [i for i in os.listdir('/kaggle/input/tomato/New Plant Diseases Dataset(Augmented)/valid/Tomato___Leaf_Mold/') if i.endswith('JPG')]:

    img = image.load_img('/kaggle/input/tomato/New Plant Diseases Dataset(Augmented)/valid/Tomato___Leaf_Mold/'+j, target_size=(256, 256))

    x = image.img_to_array(img)

    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])

    classes = model.predict(images)

    x=(list(classes[0])).index(1)

#     print(classes)

    print(j,"  ",x)
import matplotlib.pyplot as plt

history= model.history

acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend(loc=0)

plt.figure()

plt.show()

plt.savefig('graph_VAL_vs_TRAIN.png',dpi=120,quality=90,opimaize=True,)
from threading import  *

import pickle

import threading

# from multiprocessing import Queue

with open('pdsmodel.pickle', 'wb') as f:

    pickle.dump(model, f)
model.save("acc20.h5")
model=tf.keras.models.load_model('Tomatofinal.h5')
model
