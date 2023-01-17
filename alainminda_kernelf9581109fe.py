# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
labels = pd.read_csv('../input/train.csv')
import tensorflow as tf

from tensorflow.keras.optimizers import RMSprop

model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(32, 32, 3)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(64, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')

])
model.compile(optimizer = RMSprop(lr=0.0001), 

              loss = 'binary_crossentropy', 

              metrics = ['acc'])

labels['class'] = 'no_cactus'

labels.loc[labels['has_cactus'] == 1 ,['class']] = "cactus"

labels.head()

# splitting data into train and validation

from sklearn.model_selection import train_test_split

train, valid = train_test_split(labels, stratify=labels.has_cactus, test_size=0.2)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Add our data-augmentation parameters to ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255.,

                                   # rotation_range = 40,

                                   width_shift_range = 0.2,

                                   height_shift_range = 0.2,

                                   # shear_range = 0.2,

                                   zoom_range = 0.2,

                                   horizontal_flip = True)



# Note that the validation data should not be augmented!

test_datagen = ImageDataGenerator( rescale = 1.0/255. )



# Flow training images in batches of 20 using train_datagen generator

train_generator = train_datagen.flow_from_dataframe(train,

                                                    directory="../input/train/train", 

                                                    x_col='id', 

                                                    y_col='class',

                                                    batch_size = 20,

                                                    class_mode = 'binary', 

                                                    target_size = (32, 32), 

                                                    color_mode='rgb')     



# Flow validation images in batches of 20 using test_datagen generator

validation_generator =  test_datagen.flow_from_dataframe( valid,

                                                          directory="../input/train/train", 

                                                          x_col='id', 

                                                          y_col='class',

                                                          batch_size  = 50,

                                                          class_mode  = 'binary', 

                                                          target_size = (32, 32), 

                                                          color_mode='rgb')
history = model.fit_generator(

            train_generator,

            validation_data = validation_generator,

            steps_per_epoch = 100,

            epochs = 120,

            validation_steps = 50,

            verbose = 2)
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
classes = model.predict(validation_generator)

print(classes)
to_submit = pd.read_csv('../input/sample_submission.csv')
to_submit.head()
# Note that the validation data should not be augmented!

submission_datagen = ImageDataGenerator( rescale = 1.0/255. )

submission_generator =  submission_datagen.flow_from_dataframe( to_submit,

                                                          directory="../input/test/test", 

                                                          x_col='id', 

                                                          y_col=None,

                                                          shuffle=False,

                                                          batch_size  = 50,

                                                          class_mode  = None, 

                                                          target_size = (32, 32), 

                                                          color_mode='rgb')

classes = model.predict(submission_generator)

print(classes)
to_submit['has_cactus'] = classes

to_submit.head(10)
to_submit.to_csv('samplesubmission.csv',index=False)