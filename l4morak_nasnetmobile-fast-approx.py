# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from scipy.special import factorial
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from keras.models import Model
from keras.layers import *
from keras.applications import *
from keras.callbacks import *

from keras.preprocessing.image import ImageDataGenerator

image_gen = ImageDataGenerator(rotation_range = 0.5,
                               samplewise_std_normalization = True,
                               samplewise_center = True,
                               height_shift_range = 0.1,
                               width_shift_range = 0.1,
                               fill_mode = 'reflect')

data_gen = image_gen.flow_from_directory('../input/asl_alphabet_train/asl_alphabet_train/', target_size=(224, 224), batch_size = 32)

lr_schedule = LearningRateScheduler( lambda x: 0.001 / factorial(np.array([x+1]))[0], verbose = 1 ) 
checkpoint = ModelCheckpoint('output_model.hdf5', monitor = 'loss', mode = 'min', save_best_only = True, verbose = 1)
mobile_net = NASNetMobile(input_shape = (224,224,3), weights = 'imagenet', include_top = False)

inp = Input((224,224,3))
x = mobile_net(inp)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(512)(x)
x = Dropout(0.5)(x)
x = Dense(29)(x)
x = Activation('softmax')(x)

model = Model(inp, x)
model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['categorical_accuracy'])
model.fit_generator(data_gen, steps_per_epoch = 200, epochs = 10, callbacks = [lr_schedule, checkpoint], verbose = 1)