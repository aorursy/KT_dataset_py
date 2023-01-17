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
try:

  # %tensorflow_version only exists in Colab.

  %tensorflow_version 2.x

  print(tf.__version__)

except Exception:

  pass
#Import the library

%matplotlib inline



import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from pathlib import Path

import tensorflow as tf

from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.layers import (Input, Activation, Dense, Flatten, Conv2D, 

                                     MaxPooling2D, Dropout, BatchNormalization)



import numpy as np

import numpy as np

from keras.preprocessing import image

import os
def plotImages(images_arr):

    fig, axes = plt.subplots(1, 5, figsize=(20,20))

    axes = axes.flatten()

    for img, ax in zip( images_arr, axes):

        ax.imshow(img)

        ax.axis('off')

    plt.tight_layout()

    plt.show()
data_dir = Path('../input/intel-image-classification')

train_dir = data_dir / 'seg_train'/'seg_train'

val_dir = data_dir / 'seg_test'/'seg_test'

test_dir = data_dir / 'seg_pred'/'seg_pred'
mountain_dir = train_dir / 'mountain'

sea_dir = train_dir / 'sea'

forest_dir = train_dir / 'forest'

street_dir = train_dir / 'street'

glacier_dir = train_dir / 'glacier'

buildings_dir = train_dir / 'buildings'

#------------------------------------------#

mountain_val_dir = val_dir / 'mountain'

sea_val_dir = val_dir / 'sea'

forest_val_dir = val_dir / 'forest'

street_val_dir = val_dir / 'street'

glacier_val_dir = val_dir / 'glacier'

buildings_val_dir = val_dir / 'buildings'
mountain_dir_ = os.path.join(mountain_dir)

mountain_img = os.listdir(mountain_dir_)

sea_dir_ = os.path.join(sea_dir)

sea_img = os.listdir(sea_dir_)

forest_dir_ = os.path.join(forest_dir)

forest_img = os.listdir(forest_dir_)

street_dir_ = os.path.join(street_dir)

street_img = os.listdir(street_dir_)

glacier_dir_ = os.path.join(glacier_dir)

glacier_img = os.listdir(glacier_dir_)

buildings_dir_ = os.path.join(buildings_dir)

buildings_img = os.listdir(buildings_dir_)

#---––---------------------------------------#

mountain_val_dir_ = os.path.join(mountain_val_dir)

mountain_val_img = os.listdir(mountain_val_dir_)

sea_val_dir_ = os.path.join(sea_val_dir)

sea_val_img = os.listdir(sea_val_dir_)

forest_val_dir_ = os.path.join(forest_val_dir)

forest_val_img = os.listdir(forest_val_dir_)

street_val_dir_ = os.path.join(street_val_dir)

street_val_img = os.listdir(street_val_dir_)

glacier_val_dir_ = os.path.join(glacier_val_dir)

glacier_val_img = os.listdir(glacier_val_dir_)

buildings_val_dir_ = os.path.join(buildings_val_dir)

buildings_val_img = os.listdir(buildings_val_dir_)
from tensorflow.keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(rescale=1/255, 

                                   rotation_range=20,

                                   width_shift_range=0.2,

                                   height_shift_range=0.2,

                                   horizontal_flip=True,

                                  )



val_datagen = ImageDataGenerator(rescale=1/255, 

                                   rotation_range=20,

                                   width_shift_range=0.2,

                                   height_shift_range=0.2,

                                   horizontal_flip=True,

                                  )



test_datagen = ImageDataGenerator(rescale=1/255)



train_generator = train_datagen.flow_from_directory(train_dir, target_size=(300, 300), 

                                                    batch_size=128,  class_mode='categorical')



validation_generator = train_datagen.flow_from_directory(val_dir, target_size=(300, 300), 

                                                    batch_size=128,  class_mode='categorical')
input_shape=(300, 300, 3)
#Import VGG16 net

#include_top = False => Dense not included

VGG16_net = tf.keras.applications.VGG16(input_shape=input_shape, include_top=False, weights="imagenet")

VGG16_net.trainable = False

#Add top layer

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()(VGG16_net.output)

dense_1 = tf.keras.layers.Dense(units=16, activation='relu')(global_average_layer)

prediction_layer = tf.keras.layers.Dense(units=6, activation='sigmoid')(dense_1)
VGG16_net_model = tf.keras.models.Model(inputs=VGG16_net.input, outputs=prediction_layer)
#Compile the model

VGG16_net_model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])
history_model_VGG16_net = VGG16_net_model.fit_generator(train_generator, epochs=1, verbose=1)
#Plot the accuracy

plt.plot(history_model_VGG16_net.history['accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.show()
test_img_dir = os.path.join(test_dir)

test_img = os.listdir(test_dir)

print(test_img_dir)



test_path = os.path.join(test_img_dir, test_img[40]) 

img = mpimg.imread(test_path)

plt.imshow(img)
from keras.preprocessing import image
img = image.load_img(test_path, target_size=(300, 300))

x = image.img_to_array(img)

x = np.expand_dims(x, axis=0)



images = np.vstack([x])
classes = VGG16_net_model.predict(images)
print(np.argmax(classes))
train_generator.class_indices