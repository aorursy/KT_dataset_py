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
import random

from shutil import copyfile

import numpy as np

import keras

from keras import backend as K

from keras.layers.core import Dense, Activation

from keras.optimizers import Adam

from keras.metrics import categorical_crossentropy

from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing import image

from keras.models import Model

from keras.applications import imagenet_utils

from keras.layers import Dense,GlobalAveragePooling2D

from keras.applications import MobileNet

from keras.applications.mobilenet import preprocess_input

from IPython.display import Image

from keras.optimizers import Adam
IMAGE_SIZE = 224

IMAGE_DATA = '../input'



images = []

labels = []
def prepare_image(file):

    img_path = ''

    img = image.load_img(img_path + file, target_size=(IMAGE_SIZE, IMAGE_SIZE))

    img_array = image.img_to_array(img)

    img_array_expanded_dims = np.expand_dims(img_array, axis=0)

    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)
for path, subdirs, files in os.walk(IMAGE_DATA):

    for name in files:

        img_path = os.path.join(path, name)

        images.append(prepare_image(img_path))

        labels.append(path.split('/')[-1])

images = np.array(images)

# Remove single axis

images = np.squeeze(images, axis=1)
mapped_labels = list(map(lambda x: 1 if x == 'boss' else 0, labels))



from keras.utils import np_utils



y_data = np_utils.to_categorical(mapped_labels)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(images, y_data, test_size=0.3)
model = keras.applications.mobilenet.MobileNet(classes=2, weights=None)

model.summary()
#from keras.regularizers import l2

#for layer in model.layers:

#    if hasattr(layer, 'kernel_regularizer'):

#        layer.kernel_regularizer= l2(0.01)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from keras.callbacks import ModelCheckpoint



model_file = "model_mobilenet.hdf5"



checkpoint = ModelCheckpoint(model_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

callbacks_list = [checkpoint]
model.fit(x=X_train, y=y_train, batch_size=16, epochs=100, verbose=1, validation_data=(X_test, y_test), callbacks=callbacks_list)
model.save('model_non_decay.h5')
# print(os.listdir("../input/boss/boss"))

x = prepare_image('../input/boss/boss/126.png')

print(model.predict(x))
print(mapped_labels.count(0))