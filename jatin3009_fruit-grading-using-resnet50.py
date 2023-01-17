# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import os
from os import listdir, makedirs
from os.path import join, exists, expanduser

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import tensorflow as tf
img_width, img_height = 224, 224

train_data_dir = '/kaggle/input/FINAL IMAGE DATA/Train'
validation_data_dir = '/kaggle/input/FINAL IMAGE DATA/Test'
nb_train_samples = 500
nb_validation_samples = 100
batch_size = 10
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')
inception_base = applications.ResNet50(weights='imagenet', include_top=False)


x = inception_base.output
x = GlobalAveragePooling2D()(x)

x = Dense(512, activation='relu')(x)

predictions = Dense(7, activation='softmax')(x)

inception_transfer = Model(inputs=inception_base.input, outputs=predictions)
inception_base_vanilla = applications.ResNet50(weights=None, include_top=False)


x = inception_base_vanilla.output
x = GlobalAveragePooling2D()(x)

x = Dense(512, activation='relu')(x)

predictions = Dense(7, activation='softmax')(x)

inception_transfer_vanilla = Model(inputs=inception_base_vanilla.input, outputs=predictions)
inception_transfer.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

inception_transfer_vanilla.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
import tensorflow as tf
with tf.device("/device:GPU:0"):
    history_pretrained = inception_transfer.fit_generator(
    train_generator,
    epochs=90, shuffle = True, verbose = 0, validation_data = validation_generator)
    inception_transfer_vanilla.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])
with tf.device("/device:GPU:0"):
    history_vanilla = inception_transfer_vanilla.fit_generator(
    train_generator,
    epochs=90, shuffle = True, verbose = 1, validation_data = validation_generator)
import matplotlib.pyplot as plt

plt.plot(history_pretrained.history['val_accuracy'])
plt.plot(history_vanilla.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Pretrained'], loc='upper left')
plt.show()

plt.plot(history_pretrained.history['val_loss'])
plt.plot(history_vanilla.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Pretrained', 'Vanilla'], loc='upper left')
plt.show()
import os
from os import listdir, makedirs
from os.path import join, exists, expanduser
from PIL import Image
import numpy as np
from matplotlib.pyplot import imshow
from numpy import asarray

from keras.preprocessing import image

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import tensorflow as tf
%matplotlib inline
img = image.load_img('/kaggle/input/FINAL IMAGE DATA/Test/GRADE-6/image_0_5611(1).jpg', target_size=(224, 224))
imshow(np.asarray(img))

data = image.img_to_array(img)
image_batch = np.expand_dims(data, axis=0)
processed_image = preprocess_input(image_batch, mode='caffe')
pred = inception_transfer.predict(processed_image)
print(pred[0])

result_rate = pred[0].tolist()
total_class = [1,2,3,4,5,6,7]

# construct the result
result_class = dict()
for i in range(len(total_class)):
    result_class[total_class[i]] = result_rate[i]
# sort the result
result_class = sorted(result_class.items(), key=lambda x:x[1], reverse=True)
print(np.array(result_class))