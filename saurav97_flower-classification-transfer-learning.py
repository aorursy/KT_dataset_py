# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import cv2
from glob import glob
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
print(os.listdir("../input/flowers-recognition/flowers/flowers"))

labels = []
images = []
count = 0

PATH = os.path.abspath(os.path.join('..', 'input', 'flowers-recognition', 'flowers', 'flowers'))
SOURCE_IMAGES = os.path.join(PATH, "daisy")
daisy = glob(os.path.join(SOURCE_IMAGES, "*.jpg"))

for i in daisy:
    labels.append(0)
    images.append(i)
    
SOURCE_IMAGES = os.path.join(PATH, "tulip")
tulip = glob(os.path.join(SOURCE_IMAGES, "*.jpg"))

for i in tulip:
    labels.append(1)
    images.append(i)
    
SOURCE_IMAGES = os.path.join(PATH, "sunflower")
sunflower = glob(os.path.join(SOURCE_IMAGES, "*.jpg"))

for i in sunflower:
    labels.append(2)
    images.append(i)
    
SOURCE_IMAGES = os.path.join(PATH, "rose")
rose = glob(os.path.join(SOURCE_IMAGES, "*.jpg"))

for i in rose:
    labels.append(3)
    images.append(i)
    
SOURCE_IMAGES = os.path.join(PATH, "dandelion")
dandelion = glob(os.path.join(SOURCE_IMAGES, "*.jpg"))

for i in dandelion:
    labels.append(4)
    images.append(i)
Width = 227
Height = 227

x = []
y = []
n = 0

for img in images:
    base = os.path.basename(img)
    i = cv2.imread(img)
    x.append(cv2.resize(i, (Width, Height), interpolation = cv2.INTER_CUBIC))
    y.append(labels[n])
    n += 1
import numpy as np
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, BatchNormalization, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.utils import np_utils

x = np.asarray(x)
y = np.asarray(y)

x = x.astype('float32')
x = ( x - 127.5 ) / 127.5
y = np_utils.to_categorical(y, 5) 
x.shape
cache_dir = os.path.expanduser(os.path.join('~', '.keras'))
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
models_dir = os.path.join(cache_dir, 'models')
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
!cp ../input/keras-pretrained-models/inception* ~/.keras/models/
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import decode_predictions

model = InceptionV3(weights='imagenet', include_top=True, input_shape=(227, 227, 3))
for layer in model.layers:
    layer.trainable = False

new_model = Sequential()
new_model.add(model)
new_model.add(Dense(512, activation = 'relu'))
new_model.add(Dense(256, activation = 'relu'))
new_model.add(Dense(128, activation = 'relu'))
new_model.add(Dense(5, activation = 'softmax'))
new_model.summary()
new_model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
new_model.fit(x, y, batch_size=8, nb_epoch=10, verbose=1)
!ls ../input/keras-pretrained-models/
from keras.applications import inception_resnet_v2

base_model = inception_resnet_v2.InceptionResNetV2(weights = 'imagenet', include_top=False, pooling='avg')
base_model.summary()
for layer in base_model.layers:
    layer.trainable = False
model1 = Sequential()
model1.add(base_model)
model1.add(Dense(512, activation = 'relu'))
model1.add(Dense(128, activation = 'relu'))
model1.add(Dense(5, activation = 'softmax'))
model1.summary()
model1.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model1.fit(x, y, batch_size=8, nb_epoch=20, verbose=1)
