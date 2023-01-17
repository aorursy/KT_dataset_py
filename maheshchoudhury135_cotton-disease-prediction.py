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

from tensorflow.keras.layers import Flatten, Dense

from tensorflow.keras.applications import VGG19

from tensorflow.keras.preprocessing.image import ImageDataGenerator
vgg= VGG19(include_top= False, weights= "imagenet", input_shape= (224, 224, 3))
for layer in vgg.layers:

    layer.trainable = False
x= Flatten()(vgg.output)
pred= Dense(4, activation= "softmax")(x)
model= tf.keras.Model(vgg.inputs, pred)
model.compile(loss= "categorical_crossentropy", optimizer= "adam", metrics= ["acc"])
train_generator= ImageDataGenerator(rescale=1./255,  shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

test_generator= ImageDataGenerator(rescale=1./255,  shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_data = train_generator.flow_from_directory( '../input/cotton-disease-dataset/Cotton Disease/train', target_size=(224, 224),

        batch_size=64)
test_data = test_generator.flow_from_directory( '../input/cotton-disease-dataset/Cotton Disease/test', target_size=(224, 224),

        batch_size=64)
h= model.fit(train_data, steps_per_epoch= len(train_data), epochs=10, validation_data= test_data, validation_steps= len(test_data))
model_json = model.to_json()

with open("model.json", "w") as json_file:

    json_file.write(model_json)
model.save_weights("model.h5")