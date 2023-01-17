!pip install keras_vggface

!pip install tensorflow==1.13.2
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from tensorflow import keras



import keras

from keras_vggface.vggface import VGGFace



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
print(tf.version)
# pretrained_model = keras.applications.ResNet50(include_top=False, input_shape=(224, 224, 3))

pretrained_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
model = keras.models.Sequential()

model.add(pretrained_model)

# model.add(keras.layers.GlobalAveragePooling2D())

model.summary()
# Used for new models of tensorflow

# tf.saved_model.save(model, "./output/")

# converter = tf.lite.TFLiteConverter.from_saved_model("./output/")

# tflite_model = converter.convert()

# open("converted_model.tflite", "wb").write(tflite_model)
# Used for old keras models

model.save("model.h5")

converter = tf.contrib.lite.TFLiteConverter.from_keras_model_file('model.h5')

tfmodel = converter.convert()

open("model.tflite" , "wb") .write(tfmodel)