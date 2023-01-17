!pip install tensorflow-datasets
import numpy as np

import pandas as pd

import tensorflow as tf

import tensorflow_datasets as tfds



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



tf.__version__

# tfds.list_builders()

seed = 51
tf.executing_eagerly()
ds, info = tfds.load('tf_flowers:3.0.0', split='train', with_info=True)

assert isinstance(ds, tf.data.Dataset)

info
print(info.features)

print(info.features["label"].num_classes)

print(info.features["label"].names)
def decode_img(elem):

    image = elem['image']

    label = elem['label']

    # Use `convert_image_dtype` to convert to floats in the [0,1] range.

    img = tf.image.convert_image_dtype(image, tf.float32)

    # resize the image to the desired size.

    return tf.image.resize(img, [224, 224]), label
result = ds.map(decode_img)
# shuffle and slice the dataset

ds = result.shuffle(buffer_size=1024, seed=seed).batch(128)
from tensorflow.keras.layers import Input, Dense, Conv2D, ELU, MaxPooling2D, Flatten, SeparableConv2D

from tensorflow.keras.models import Model



inputs = Input(shape=(224, 224, 3), name='flowers')

x = Conv2D(filters=64, kernel_size=(5, 5), padding='valid', use_bias=False)(inputs)

x = ELU()(x)

x = MaxPooling2D()(x)



x = Conv2D(filters=64, kernel_size=(3, 3), padding='valid', use_bias=False)(x)

x = ELU()(x)

x = MaxPooling2D()(x)



x = Conv2D(filters=128, kernel_size=(3, 3), padding='valid', use_bias=False)(x)

x = ELU()(x)

x = MaxPooling2D()(x)



x = Conv2D(filters=256, kernel_size=(3, 3), padding='valid', use_bias=False)(x)

x = ELU()(x)

x = MaxPooling2D()(x)



x = Conv2D(filters=512, kernel_size=(3, 3), padding='valid', use_bias=False)(x)

x = ELU()(x)

x = MaxPooling2D()(x)



x = SeparableConv2D(512, kernel_size=3, padding='valid')(x)

x = ELU()(x)



x = SeparableConv2D(512, kernel_size=3, padding='valid')(x)

x = ELU()(x)



x = Flatten()(x)



x = Dense(128, activation='relu', name='dense_1')(x)

x = Dense(64, activation='relu', name='dense_2')(x)

x = Dense(32, activation='relu', name='dense_3')(x)

outputs = Dense(5, activation='softmax', name='predictions')(x)



model = Model(inputs=inputs, outputs=outputs)

model.summary()
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.losses import SparseCategoricalCrossentropy

from tensorflow.keras.metrics import SparseCategoricalAccuracy



model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=[SparseCategoricalAccuracy()])
history = model.fit(ds, epochs=20, verbose=1).history
import matplotlib.pyplot as plt

import seaborn as sns



fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', figsize=(20, 14))



ax1.plot(history['loss'], label='Train loss')

# ax1.plot(history['val_loss'], label='Validation loss')

ax1.legend(loc='best')

ax1.set_title('Loss')



ax2.plot(history['sparse_categorical_accuracy'], label='Train accuracy')

# ax2.plot(history['val_acc'], label='Validation accuracy')

ax2.legend(loc='best')

ax2.set_title('Accuracy')



plt.xlabel('Epochs')

sns.despine()

plt.show()