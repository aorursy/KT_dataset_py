!pip install -U tensorflow

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.data.experimental.ops import interleave_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.tpu import datasets
import numpy as np
import pandas as pd
#trainImgDir = KaggleDatasets().get_gcs_path('bengali-sign-language-dataset/RESIZED_DATASET')
trainImgDir = '/kaggle/input/bengali-sign-language-dataset/RESIZED_DATASET/'
IMG_DIM = 224
new_input = tf.keras.Input(shape=(IMG_DIM, IMG_DIM,3))
model = tf.keras.applications.VGG19(weights='imagenet', include_top=False, input_tensor=new_input, input_shape=(224, 224, 3))
for layer in model.layers:
    layer.trainable = False
flat1 = tf.keras.layers.Flatten()(model.layers[-2].output)
class1 = tf.keras.layers.Dense(256, activation='relu')(flat1)
drop = tf.keras.layers.Dropout(.3) (class1) 
output = tf.keras.layers.Dense(38, activation='softmax')(drop)

model = tf.keras.Model(inputs=model.inputs, outputs=output)
model.summary()

data_augmentor = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255.0,samplewise_center=True, 
                                    rotation_range=30,zoom_range=0.15,width_shift_range=0.2,height_shift_range=0.2,
                                    shear_range=0.15,horizontal_flip=True, vertical_flip = True,
                                    validation_split=0.1)

train_generator = data_augmentor.flow_from_directory(directory = '/kaggle/input/bengali-sign-language-dataset/RESIZED_DATASET/', target_size=(224, 224), batch_size=16,
                                                     shuffle=True, class_mode="categorical", seed=42, subset="training")
valid_generator = data_augmentor.flow_from_directory(directory = '/kaggle/input/bengali-sign-language-dataset/RESIZED_DATASET/', target_size=(224, 224), batch_size=16, 
                                                   shuffle=True, class_mode="categorical", seed=42, subset="validation")
train_generator.dtype
a = tf.data.Dataset.from_generator(lambda: train_generator,output_types=(tf.float32, tf.float32),
    output_shapes = ([None,224,224,3],[None,38]))
#list(a.take(3).as_numpy_iterator())
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='loss', factor=.2, patience=5, verbose=0, mode='auto',
   min_lr=0.0001
)
terminate_NAN = tf.keras.callbacks.TerminateOnNaN()
early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
callback = ['reduce_lr', 'early_stop']
opt = tf.keras.optimizers.SGD(lr=0.001)
model.compile(optimizer=opt, loss=['categorical_crossentropy'], metrics=['accuracy'])

model.fit(tf.data.Dataset.from_generator(lambda: train_generator,output_types=(tf.float32, tf.float32),
    output_shapes = ([None,224,224,3],[None,38])), steps_per_epoch=train_generator.n // train_generator.batch_size,
           validation_data=tf.data.Dataset.from_generator(lambda: valid_generator,output_types=(tf.float32, tf.float32),
    output_shapes = ([None,224,224,3],[None,38])), validation_steps=valid_generator.n // valid_generator.batch_size,
           epochs=100, callbacks = [reduce_lr, terminate_NAN, early_stop]).history
print ("XXX")
