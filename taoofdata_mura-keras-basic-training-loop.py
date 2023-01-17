# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import tensorflow as tf

from tensorflow.keras import layers

from skimage.io import imshow

from pathlib import Path

import pandas as pd



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
dataset_root = Path('../input/mura/MURA-v1.1')
list(dataset_root.iterdir())
df = pd.read_csv(dataset_root/'train_image_paths.csv', header=None, names=['filename'])

df.head()
df['class'] = (df.filename

               .str.extract('study.*_(positive|negative)'))

df.head()
def generate_df(dataset_root, csv_name):

    df = pd.read_csv(dataset_root/csv_name, header=None, names=['filename'])

    df['class'] = (df.filename

               .str.extract('study.*_(positive|negative)'))

    return df
list(dataset_root.parent.iterdir())
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1. / 255)

train_gen = datagen.flow_from_dataframe(generate_df(dataset_root, 'train_image_paths.csv'),

                                        directory=dataset_root.parent,

                                        target_size=(224, 224),

                                        class_mode='binary')

valid_gen = datagen.flow_from_dataframe(generate_df(dataset_root, 'valid_image_paths.csv'),

                                        directory=dataset_root.parent,

                                        target_size=(224, 224),

                                        class_mode='binary')
densenet = tf.keras.applications.DenseNet169(weights='imagenet', include_top = False, input_shape=(224, 224, 3))
densenet.trainable = False
densenet.summary()
model = tf.keras.models.Sequential([

    densenet,

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(1, activation='sigmoid')

])
model.summary()
model.compile(loss=tf.keras.losses.binary_crossentropy,

              optimizer=tf.keras.optimizers.Adam(),

              metrics=['accuracy'])
model.fit_generator(train_gen, epochs=5, validation_data=valid_gen, use_multiprocessing=True)