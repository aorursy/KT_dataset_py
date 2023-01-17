# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/10-monkey-species/"))

# Any results you write to the current directory are saved as output.
import os, cv2
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import img_to_array, array_to_img, load_img
from keras.utils import np_utils

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')
img_width = 150
img_height = 150

data_dir = '../input/flowers-recognition/flowers/flowers/'

images = []
flower_types = []
for flower_type in os.listdir(data_dir):
    flower_dir = data_dir + flower_type
    flower_files = [flower_dir + '/' + filename for filename in os.listdir(flower_dir)]
    for filename in flower_files:
        if filename.endswith('jpg'):
            images.append(cv2.resize(cv2.imread(filename), (img_width,img_height), interpolation=cv2.INTER_CUBIC))
            flower_types.append(flower_type)
            
images = np.array(images)
flower_types = np.array(flower_types)

le = LabelEncoder()
flower_types_encoded = le.fit_transform(flower_types)
flower_types_encoded_onehot = np_utils.to_categorical(flower_types_encoded)
'''
_, ax = plt.subplots(5,6, figsize=(30,30))
for i in range(5):
    for j in range(6):
      ax[i,j].imshow(cv2.cvtColor(images[(i*548)+j], cv2.COLOR_BGR2RGB))
      ax[i,j].axis('off')
      ax[i,j].set_title(le.inverse_transform(flower_types_encoded[(i*548)+j]))
'''

from sklearn.utils import class_weight
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
import keras
map_characters1 = {0: 'daisy', 1: 'rose',2:'tulip',3:'dandelion',4:'sunflower'}
class_weight1 = class_weight.compute_class_weight('balanced', np.unique(flower_types), flower_types)
weight_path1 = '../input/keras-pretrained-models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
weight_path2 = '../input/keras-pretrained-models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
pretrained_model_1 = VGG16(weights = weight_path1, include_top=False, input_shape=(150, 150, 3))
pretrained_model_2 = InceptionV3(weights = weight_path2, include_top=False, input_shape=(150, 150, 3))
optimizer1 = keras.optimizers.RMSprop(lr=0.0001)
vgg16_layers_to_freeze = 11
inception_layers_to_freeze = 172

from keras.optimizers import SGD

def pretrainedNetwork(data,lab,pretrainedmodel,pretrainedweights,classweight,numclasses,numepochs,optimizer,labels,fine_tuned=False,layers=None):
    base_model = pretrained_model_1 # Topless
    # Add top layer
    x = base_model.output
    x = Flatten()(x)
    predictions = Dense(numclasses, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    if fine_tuned:
        # fine tuning
        for layer in base_model.layers[:layers]:
            layer.trainable = False
        for layer in base_model.layers[layers:]:
            layer.trainable = True
        # note the optimiser: the learning rate should be lower than the model was trained with
        #model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    else:
        # transfer learning
        for layer in base_model.layers:
            layer.trainable = False
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
       # model.copile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
    callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=1)]
    model.summary()
    # Fit model
    history = model.fit(data,lab, epochs=numepochs, class_weight=classweight, validation_split=0.2 , verbose=1)
    # Evaluate model
    return model
pretrainedNetwork(images,flower_types_encoded_onehot,pretrained_model_2,weight_path2,class_weight1,5,15,optimizer1,map_characters1)
images, flower_types, flower_types_encoded = shuffle(images, flower_types, flower_types_encoded)

from keras.applications.vgg16 import VGG16
weights = '../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
vgg16_model = VGG16(include_top=False, weights=weights)

from os import makedirs
from os.path import join, exists, expanduser

cache_dir = expanduser(join('~', '.keras'))
if not exists(cache_dir):
    makedirs(cache_dir)
models_dir = join(cache_dir, 'models')
if not exists(models_dir):
    makedirs(models_dir)
!cp  ../input/vgg16/* ~/.keras/models/

bottleneck_features = vgg16_model.predict(images)

input_shape = bottleneck_features.shape[1:]
num_classes = flower_types_encoded_onehot.shape[1]

model_top = Sequential()
model_top.add(Flatten(input_shape=input_shape))
model_top.add(Dense(256, activation='relu'))
model_top.add(Dense(128, activation='relu'))
model_top.add(Dense(64, activation='relu'))
model_top.add(Dropout(0.5))
model_top.add(Dense(num_classes, activation='sigmoid'))

model_top.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

print (model_top.summary())



history = model_top.fit(bottleneck_features, 
                        flower_types_encoded_onehot,
                        epochs=30,
                        batch_size=20,
                        validation_split=0.25)