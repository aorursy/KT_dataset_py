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
train_data_path="../input/intel-image-classification/seg_train/seg_train"
test_data_path="../input/intel-image-classification/seg_test/seg_test"
import matplotlib.pyplot as plt
%matplotlib inline
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_gen=ImageDataGenerator(rotation_range=40,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,vertical_flip=True,rescale=1/255)
train_generator=train_gen.flow_from_directory(train_data_path,target_size=(150,150),batch_size=64,class_mode='categorical')

test_gen=ImageDataGenerator(rescale=1/255)
test_generator=test_gen.flow_from_directory(test_data_path,target_size=(150,150),batch_size=64,class_mode='categorical')
train_num = train_generator.samples
validation_num = test_generator.samples 
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=(150,150,3),activation='relu'))
model.add(MaxPooling2D(2,2))
Dropout(.2)
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
Dropout(.2)
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
Dropout(.2)
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(6,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

early_stop=EarlyStopping(patience=3,monitor='loss')
model.fit_generator(train_generator,epochs=50,validation_data=test_generator
                    ,callbacks=[early_stop])

