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
from keras.layers import Input,Lambda,Dense,Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16,preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

IMAGE_SIZE=[224,224]
data_dir = '../input/chest-xray-pneumonia/chest_xray/'

# Path to train directory (Fancy pathlib...no more os.path!!)
train_dir = data_dir + 'train'

# Path to validation directory
val_dir = data_dir + 'val'

# Path to test directory
test_dir = data_dir + 'test'
vgg=VGG16(input_shape=IMAGE_SIZE+[3],weights='imagenet',include_top=False)
for layer in vgg.layers:
    layer.trainable = False
x=Flatten()(vgg.output)
folders=glob(train_dir+'/*')
len(folders)
prediction=Dense(len(folders),activation='softmax')(x)
model=Model(inputs=vgg.input,outputs=prediction)
model.summary()
from sklearn.metrics import recall_score
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
from keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
valid_datagen=ImageDataGenerator(rescale=1./255)
test_datagen=ImageDataGenerator(rescale=1./255)


training_set=train_datagen.flow_from_directory('../input/chest-xray-pneumonia/chest_xray/train',target_size=(224,224),batch_size=32,class_mode='categorical')
val_set=valid_datagen.flow_from_directory('../input/chest-xray-pneumonia/chest_xray/val',target_size=(224,224),batch_size=32,class_mode='categorical')
test_set=test_datagen.flow_from_directory('../input/chest-xray-pneumonia/chest_xray/test',target_size=(224,224),batch_size=32,class_mode='categorical')
r=model.fit_generator(training_set,validation_data=test_set,epochs=2,steps_per_epoch=len(training_set),validation_steps=len(test_set))
r.history.keys()
feature=model.predict(test_set)
feature
feature.shape
preds = np.argmax(feature, axis=-1)
