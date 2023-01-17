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
from keras.preprocessing.image import load_img
img=load_img("/kaggle/input/intel-image-classification/seg_pred/seg_pred/16040.jpg")
img
img.size
import os
from os import listdir
listdir('/kaggle/input')
listdir('/kaggle/input/intel-image-classification')
listdir('/kaggle/input/intel-image-classification/seg_train/')
listdir('/kaggle/input/intel-image-classification/seg_train/seg_train')
listdir('/kaggle/input/intel-image-classification/seg_train/seg_train/buildings')
x_train, y_train = list(), list()

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
folder='/kaggle/input/intel-image-classification/seg_train/seg_train'
for file1 in listdir(folder):
    file2=folder+'/'+file1
    for file3 in listdir(file2):
        file4=file2+'/'+file3
        image = load_img(file4,target_size=(150,150))
        image=img_to_array(image)
        x_train.append(image)
        y_train.append(file1)
        
from numpy import asarray

x_train=asarray(x_train)
y_train=asarray(y_train)
x_train.shape
y_train.shape
x_test, y_test = list(), list()
folder='/kaggle/input/intel-image-classification/seg_test/seg_test'
for file1 in listdir(folder):
    file2=folder+'/'+file1
    for file3 in listdir(file2):
        file4=file2+'/'+file3
        image = load_img(file4,target_size=(150,150))
        image=img_to_array(image)
        x_test.append(image)
        y_test.append(file1)
        
x_test=asarray(x_test)
y_test=asarray(y_test)
x_test.shape
y_test.shape
# baseline cnn model for fashion mnist
from numpy import mean
from numpy import std
from matplotlib import pyplot
from sklearn.model_selection import KFold
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
import numpy
numpy.unique(y_train)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
y_train=le.fit_transform(y_train)
numpy.unique(y_train)
y_test=le.fit_transform(y_test)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
train_norm = x_train.astype('float32')
test_norm = x_test.astype('float32')
train_norm = train_norm / 255.0
test_norm = test_norm / 255.0
# example of loading the vgg16 model
from keras.applications.vgg16 import VGG16
# load model without classifier layers
model = VGG16(include_top=False, input_shape=(150, 150, 3))
from keras.models import Model
# define cnn model
def define_model():
# load model
  model = VGG16(include_top=False, input_shape=(150, 150, 3))
# mark loaded layers as not trainable
  for layer in model.layers:
    layer.trainable = False
# add new classifier layers
  flat1 = Flatten()(model.layers[-1].output)
  class1 = Dense(128, activation="relu", kernel_initializer="he_uniform")(flat1)
  output = Dense(6, activation="softmax")(class1)
# define new model
  model = Model(inputs=model.inputs, outputs=output)
# compile model
  opt = SGD(lr=0.001, momentum=0.9)
  model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
  return model
k1=define_model()
k1.fit(train_norm, y_train, epochs=5, batch_size=64, validation_data=(test_norm, y_test), verbose=0)
_, acc =k1.evaluate(test_norm, y_test, verbose=0)
acc
