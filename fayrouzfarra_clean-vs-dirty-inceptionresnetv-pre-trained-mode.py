# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
! unzip ../input/platesv2/plates.zip
print(os.listdir("/kaggle/working/"))
! ls plates
import tensorflow as tf 

tf.__version__
from keras.models import Sequential, Model

from keras.layers import Dense, Conv2D, Flatten, Activation, Dropout

from keras.preprocessing.image import ImageDataGenerator

from keras import applications, optimizers
train_image_generator=ImageDataGenerator(

        rescale=1./255,

        rotation_range=40,

        width_shift_range=0.2,

        height_shift_range=0.2,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True,

        vertical_flip = True

        )
img_size = 200

batch_size = 32
train_generator = train_image_generator.flow_from_directory(

        'plates/train',

        target_size=(img_size, img_size),

        batch_size=batch_size,

        class_mode='binary')
test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow_from_directory(  

        'plates',

        classes=['test'],

        target_size = (img_size, img_size),

        batch_size = 1,

        shuffle = False,        

        class_mode = None)
def plotImages(images_arr):

    fig, axes = plt.subplots(1, 5, figsize=(20,20))

    axes = axes.flatten()

    for img, ax in zip( images_arr, axes):

        ax.imshow(img)

        ax.axis('off')

    plt.tight_layout()

    plt.show()
import os

import numpy as np

import matplotlib.pyplot as plt

sample_training_images, _ = next(train_generator)

plotImages(sample_training_images[:5])
IMG_SHAPE=(img_size, img_size, 3)

base_model = tf.keras.applications.InceptionResNetV2(

input_shape=IMG_SHAPE,include_top=False,weights='imagenet'

        )
base_model.trainable = False
base_model.summary()
image_batch = sample_training_images[:batch_size]

image_batch.shape
feature_batch = base_model(image_batch)

feature_batch.shape
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

feature_batch_average = global_average_layer(feature_batch)

print(feature_batch_average.shape)
prediction_layer = tf.keras.layers.Dense(1)

prediction_batch = prediction_layer(feature_batch_average)

print(prediction_batch.shape)
model = tf.keras.Sequential([

  base_model,

  global_average_layer,

  prediction_layer

])
model.compile(loss='binary_crossentropy',

              optimizer = optimizers.RMSprop(lr=0.0001),

              metrics=['accuracy'])
model.summary()
initial_epochs = 20

es = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=5)

checkpoint_filepath = 'checkpoint.h5'

mc = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, monitor='accuracy', mode='max', verbose=1, save_best_only=True)

history = model.fit(train_generator,

                    epochs=initial_epochs,

                    callbacks=[es, mc])
base_model.trainable = True
print("Number of layers in the base model: ", len(base_model.layers))



# Fine-tune from this layer onwards

fine_tune_at = 100



# Freeze all the layers before the `fine_tune_at` layer

for layer in base_model.layers[:fine_tune_at]:

  layer.trainable =  False
base_learning_rate=0.0001

model.compile(loss=tf.losses.BinaryCrossentropy(from_logits=True),

              optimizer = tf.optimizers.RMSprop(lr=base_learning_rate/10),

              metrics=[tf.metrics.BinaryAccuracy(threshold=0.0, name='accuracy')])
model.summary()
fine_tune_epochs = 20

total_epochs =  initial_epochs + fine_tune_epochs



history_fine = model.fit(train_generator,

                         epochs=total_epochs,

                         initial_epoch =  history.epoch[-1],

                         callbacks=[es, mc])
saved_model = tf.keras.models.load_model('checkpoint.h5')
test_generator.reset()

predict = saved_model.predict_generator(test_generator, steps = len(test_generator.filenames))

len(predict)
sub_df = pd.read_csv('../input/platesv2/sample_submission.csv')

sub_df.head()
sub_df['label'] = predict

sub_df['label'] = sub_df['label'].apply(lambda x: 'dirty' if x > 0.5 else 'cleaned')

sub_df.head()
sub_df['label'].value_counts()
sub_df.to_csv('sub.csv', index=False)