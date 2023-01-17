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
import cv2
train_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1/255.0)
train_data = train_gen.flow_from_directory(
    r"../input/cat-and-dog/training_set/training_set/",
    batch_size = 32,
    target_size = (300, 300),
    class_mode = 'binary'
)
test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1/255.0)
test_data = test_gen.flow_from_directory(
    r"../input/cat-and-dog/test_set/test_set/",
    target_size = (300, 300),
    batch_size = 32,
    class_mode = 'binary')
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation = 'relu', input_shape = (300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(rate = 0.5),
    
    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(rate = 0.5),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(rate = 0.5),

    tf.keras.layers.Conv2D(98, (3, 3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(rate = 0.5),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation = 'relu'),
    tf.keras.layers.Dense(128, activation = 'relu'),
    
    tf.keras.layers.Dense(1, activation = 'sigmoid')  
])
model.summary()
model.compile(loss= 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
model.fit(train_data, epochs = 10, steps_per_epoch = 251)
model.evaluate(test_data)
omg = cv2.imread('../input/cat-and-dog/test_set/test_set/cats/cat.4029.jpg', 3)
omg = cv2.resize(omg, (300, 300))
omg = np.expand_dims(omg, axis=0)
omg.shape
pred = model.predict(omg)
print(pred[0, 0])
