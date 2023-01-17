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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
batch_size=64
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    zoom_range=(0.95, 0.95),
    horizontal_flip=True,
    vertical_flip=False,
    data_format="channels_last",
    dtype=tf.float32,
)
ds_train = datagen.flow_from_directory(
    "../input/10-monkey-species/training/training",
    target_size=(224, 224),
    batch_size=batch_size,
    color_mode="rgb",
    class_mode="sparse",
    shuffle=True,
    seed=123,
)
ds_valid = datagen.flow_from_directory(
    "../input/10-monkey-species/validation/validation",
    target_size=(224, 224),
    batch_size=batch_size,
    color_mode="rgb",
    class_mode="sparse",
    shuffle=True,
    seed=123,
)
model = keras.applications.InceptionV3(include_top=True)
model.trainable=False
inputs=model.layers[0].input
out=model.layers[-2].output
out=layers.Dropout(0.5)(out)
outputs=layers.Dense(10)(out)
final_model = keras.Model(inputs,outputs)
final_model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True)],
    metrics=["accuracy"],
)

history=final_model.fit(ds_train, epochs=10 , validation_data=ds_valid)
from matplotlib import pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
