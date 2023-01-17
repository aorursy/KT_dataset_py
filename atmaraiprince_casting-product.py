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
import os
from os import listdir
listdir("/kaggle/input")
listdir("/kaggle/input/real-life-industrial-dataset-of-casting-product")
listdir("/kaggle/input/real-life-industrial-dataset-of-casting-product/casting_data/casting_data")
listdir("/kaggle/input/real-life-industrial-dataset-of-casting-product/casting_data/casting_data/train")
listdir("/kaggle/input/real-life-industrial-dataset-of-casting-product/casting_data/casting_data/train/def_front")
from PIL import Image
img = Image.open("/kaggle/input/real-life-industrial-dataset-of-casting-product/casting_data/casting_data/train/def_front/cast_def_0_3040.jpeg")
img
from PIL import Image
img = Image.open("/kaggle/input/real-life-industrial-dataset-of-casting-product/casting_data/casting_data/train/def_front/cast_def_0_3021.jpeg")
img
from PIL import Image
img = Image.open("/kaggle/input/real-life-industrial-dataset-of-casting-product/casting_data/casting_data/train/def_front/cast_def_0_5595.jpeg")
img
listdir("/kaggle/input/real-life-industrial-dataset-of-casting-product/casting_data/casting_data/train/ok_front")
from PIL import Image
img = Image.open("/kaggle/input/real-life-industrial-dataset-of-casting-product/casting_data/casting_data/train/ok_front/cast_ok_0_2038.jpeg")

img
from keras.preprocessing.image import load_img
img=load_img("/kaggle/input/real-life-industrial-dataset-of-casting-product/casting_data/casting_data/train/ok_front/cast_ok_0_2038.jpeg")
img
img.size
import numpy as np
img=np.array(img)
img.shape

from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator()
# load and iterate training dataset
train_it = datagen.flow_from_directory('/kaggle/input/real-life-industrial-dataset-of-casting-product/casting_data/casting_data/train', class_mode='binary',target_size=(256,256),batch_size=64)
# load and iterate test dataset
test_it = datagen.flow_from_directory('/kaggle/input/real-life-industrial-dataset-of-casting-product/casting_data/casting_data/test', class_mode='binary',target_size=(256,256), batch_size=64)
train_it.class_indices

from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout
model=Sequential()
def ima():
  model=Sequential()
  model.add(Conv2D(10,(3,3),padding="same",activation="relu",input_shape=(256,256, 3)))
  model.add(MaxPooling2D((2,2),strides=(2,2)))
  model.add(Flatten())
  model.add(Dense(16, activation="relu", kernel_initializer="he_uniform"))
  model.add(Dense(1, activation="sigmoid"))
  opt = Adam(lr=0.01)
  model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
  return model 

model=ima()
model.summary()
results = model.fit_generator(train_it,epochs=20,validation_data=test_it,verbose=1)
import matplotlib.pyplot as plt
from matplotlib import pyplot
pyplot.title("Classification Accuracy")
pyplot.plot(results.history["accuracy"], color="blue", label="train")
pyplot.plot(results.history["val_accuracy"], color="orange", label="test")


from keras import Model
## example of loading the vgg16 model
from keras.applications.vgg16 import VGG16

# define cnn model
def define_model():
    model = VGG16(include_top=False, input_shape=(256,256, 3))
    #mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False

    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(64, activation="relu", kernel_initializer="he_uniform")(flat1)
    class1 = Dense(32, activation="relu", kernel_initializer="he_uniform")(class1)
    class1 = Dense(16, activation="relu", kernel_initializer="he_uniform")(class1)
    output = Dense(1, activation="sigmoid")(class1)
# define new model
    model = Model(inputs=model.inputs, outputs=output)
# compile model
    opt = Adam(lr=0.01)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    return model
model1=define_model()
model1.summary()
results = model1.fit_generator(train_it,epochs=5,validation_data=test_it,verbose=1)
pyplot.title("Classification Accuracy")
pyplot.plot(results.history["accuracy"], color="blue", label="train")
pyplot.plot(results.history["val_accuracy"], color="orange", label="test")
