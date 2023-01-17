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
# ../input/face-emotion/train/train
# ../input/face-emotion/val/val

import keras

from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
img_w = 48
img_h = 48
img_l = 1
batch_size = 50
kernal_size = (3,3)
input_layer = Input(shape=(img_w,img_h,img_l))
# Layer 1
l1 = Conv2D(32,kernel_size=kernal_size,activation='relu')(input_layer)
l2 = BatchNormalization()(l1)
l3 = MaxPooling2D(pool_size=(2,2))(l2)
l4 = Dropout(0.2)(l3)

# Layer 2
l5 = Conv2D(64,kernel_size=kernal_size,activation='relu')(l4)
l6 = BatchNormalization()(l5)
l7 = MaxPooling2D(pool_size=(2,2))(l6)
l8 = Dropout(0.2)(l7)

# Layer 3
l9 = Conv2D(128,kernel_size=kernal_size,activation='relu')(l8)
l10 = BatchNormalization()(l9)
l11 = MaxPooling2D(pool_size=(2,2))(l10)
l12 = Dropout(0.2)(l11)

# Layer 4
l13 = Conv2D(512,kernel_size=kernal_size,activation='relu')(l12)
l14 = BatchNormalization()(l13)
l15 = MaxPooling2D(pool_size=(2,2))(l14)
l16 = Dropout(0.2)(l15)

# # Layer 5
# l17 = Conv2D(512,kernel_size=kernal_size,activation='relu')(l16)
# l18 = BatchNormalization()(l17)
# l19 = MaxPooling2D(pool_size=(3,3))(l18)
# l20 = Dropout(0.2)(l19)
l21 = Flatten()(l16)

# Fully Connected Layers
l22 = Dense(256 , activation='relu')(l21)
l23 = BatchNormalization()(l22)
l24 = Dropout(0.25)(l23)

l25 = Dense(512 , activation='relu')(l24)
l26 = BatchNormalization()(l25)
l27 = Dropout(0.25)(l26)
nb_classes = 7

output = Dense(nb_classes,activation='softmax')(l27)
model = keras.models.Model([input_layer],output)
model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
    loss=keras.losses.categorical_crossentropy,
    metrics=['accuracy'])
# ../input/face-emotion/train/train
# ../input/face-emotion/val/val
from keras.preprocessing.image import ImageDataGenerator
train = ImageDataGenerator(rescale=1./255)
val = ImageDataGenerator(rescale=1./255)
train = train.flow_from_directory('../input/face-emotion/train/train',target_size=(img_w,img_h),batch_size=batch_size)
validation = val.flow_from_directory('../input/face-emotion/val/val',target_size=(img_w,img_h),batch_size=batch_size)
train_step_epoch = 570 # 28709 = step * batch
val_step_epoch = 71 # 3589 = step * batch
model.summary()
results = model.fit(
    train,
    validation_data=validation,
    steps_per_epoch=train_step_epoch,
    validation_steps=val_step_epoch,
    epochs = 10)
