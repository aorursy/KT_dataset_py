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
import os

import pandas as pd

import numpy as np

import tensorflow as tf

from tensorflow import keras

from keras.models import Sequential, Model

from keras.layers import Dense, Conv2D, Flatten, Activation, Dropout

from keras.preprocessing.image import ImageDataGenerator

from keras import applications, optimizers
!unzip -q ../input/platesv2/plates.zip
!ls
img_size = 200

batch_size = 32




train_datagen=ImageDataGenerator(

#         rotation_range=40,

#         width_shift_range=0.2,

#         height_shift_range=0.2,

#         shear_range=0.2,

#         zoom_range=0.2,

#         horizontal_flip=True,

#         vertical_flip = True

        )



train_generator = train_datagen.flow_from_directory(

        'plates',

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
base_model = applications.InceptionResNetV2(weights='imagenet', 

                          include_top=False, 

                          input_shape=(img_size, img_size, 3))
base_model.trainable = False


inputs = keras.Input(shape=(150, 150, 3))

# We make sure that the base_model is running in inference mode here,

# by passing `training=False`. This is important for fine-tuning, as you will

# learn in a few paragraphs.

x = base_model(inputs, training=False)

# Convert features of shape `base_model.output_shape[1:]` to vectors

x = keras.layers.GlobalAveragePooling2D()(x)

# A Dense classifier with a single unit (binary classification)

outputs = keras.layers.Dense(1)(x)

model = keras.Model(inputs, outputs)
# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

es = keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10)

# mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

mc = keras.callbacks.ModelCheckpoint('best_model.h5', monitor='accuracy', mode='max', verbose=1, save_best_only=True)





model.compile(optimizer=keras.optimizers.Adam(),

              loss=keras.losses.BinaryCrossentropy(from_logits=True),

              metrics=[keras.metrics.BinaryAccuracy()])

model.fit(train_generator, epochs=10, callbacks=[es, mc] )
test_generator.reset()

predict = model.predict_generator(test_generator, steps = len(test_generator.filenames))

len(predict)
sub_df = pd.read_csv('../input/platesv2/sample_submission.csv')

sub_df.head()
sub_df['label'] = predict

sub_df['label'] = sub_df['label'].apply(lambda x: 'dirty' if x > 0.5 else 'cleaned')

sub_df.head()
sub_df['label'].value_counts()
sub_df.to_csv('sub.csv', index=False)