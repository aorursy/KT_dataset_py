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
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, Input, MaxPooling2D
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model

import pandas as pd
import numpy as np
import gc
import cv2 as cv
from tqdm import tqdm
from sklearn.utils import shuffle
all_images = os.listdir('../input/train')
cat = [x for x in all_images if(x.split('.')[0]=='cat')]
dog = [x for x in all_images if(x.split('.')[0]=='dog')]
(img_width, img_height) = (200, 200)
input_shape = (img_width, img_height, 3)
def get_images(files):
    images = []
    labels = []
    #cat = 0, dog = 1
    for img in tqdm(files):
        if 'cat' in img:
            labels.append(0)
        else:
            labels.append(1)

        path = os.path.join('../input/train', img)    
        i = cv.imread(path)
        i = cv.resize(i, (img_width, img_height))
        images.append(i)
    return np.array(images), labels
train = cat[:1000]
train.extend(dog[:1000])
train_images, train_labels = get_images(train)

val = cat[1000:1300]
val.extend(dog[1000:1300])
val_images, val_labels = get_images(val)
train_images, train_labels = shuffle(train_images, train_labels, random_state = 0)
# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2,
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True, rotation_range = 20)

val_datagen = ImageDataGenerator(rescale=1./255)
train_datagen.fit(train_images)
val_datagen.fit(val_images)

def conv2d(last_layer, kernels, drop=True, drop_rate=0.2):
    x = Conv2D(kernels, (3, 3), activation='relu')(last_layer)
    x = Conv2D(kernels, (3, 3), activation='relu')(x)
    x = MaxPooling2D(strides=(2,2))(x)
    if drop:
        x = Dropout(drop_rate)(x)
        return x
    return x
input1 = Input(shape=input_shape)
x = conv2d(input1, 32)
x = conv2d(x, 64)
# x = conv2d(x, 128)
flat = Flatten()(x)
dense = Dense(32, activation='relu')(flat)
out = Dense(1, activation='sigmoid')(dense)

model = Model(inputs = [input1], outputs = [out])
model.summary()
# opt = optimizers.SGD(lr=0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
# opt = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
opt = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
model.compile(opt, loss = 'binary_crossentropy', metrics=['accuracy'])
epoch = 50
batch_size = 8*2
model.fit_generator(train_datagen.flow(train_images, train_labels, batch_size=batch_size),                     
                    steps_per_epoch=len(train_images) / batch_size, epochs=epoch,
                    validation_data=val_datagen.flow(val_images, val_labels, batch_size=batch_size), 
                    validation_steps = len(val_images)/batch_size)
