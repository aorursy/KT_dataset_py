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
import matplotlib.pyplot as plt
df_train = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/train.csv')
df_test = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/test.csv')

train_images = []
test_images = []

for filename in os.listdir('/kaggle/input/plant-pathology-2020-fgvc7/images/'):
    if 'Train' in filename:
        img = tf.keras.preprocessing.image.load_img('/kaggle/input/plant-pathology-2020-fgvc7/images/{}'.format(filename), target_size = (224,224, 3))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = img/255
        train_images.append(img)
    else:
        img = tf.keras.preprocessing.image.load_img('/kaggle/input/plant-pathology-2020-fgvc7/images/{}'.format(filename), target_size = (224,224, 3))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = img/255
        test_images.append(img)
train_images = np.array(train_images)
test_images = np.array(test_images)
# plt.imshow(train_images[0])
# plt.show()


plt.imshow(test_images[0])
plt.show()
from keras.applications import InceptionV3
conv_base = InceptionV3(weights = 'imagenet', include_top = False, input_shape = (224, 224, 3))
conv_base.summary()
from keras import layers, models
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(units = 256, activation = 'relu'))
model.add(layers.Dense(units = 128, activation = 'relu'))
model.add(layers.Dense(units = 4, activation = 'softmax'))
model.summary()
print(len(model.trainable_weights))
conv_base.trainable = False
print(len(model.trainable_weights))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
train_labels = np.array(df_train.drop(['image_id'], axis = 1))
train_labels
model.fit(train_images, train_labels, epochs = 10, verbose = True, batch_size = 64, validation_split = 0.1)
df_train.head(10)
