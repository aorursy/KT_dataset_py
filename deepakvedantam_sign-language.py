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
import numpy as np
from tensorflow import keras
from keras.layers import Dense,Flatten,Conv2D,MaxPool2D,Dropout
from keras.models import Sequential
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

train=ImageDataGenerator(rescale=1/255,validation_split=0.02)
train_dataset=train.flow_from_directory('../input/asl-alphabet/asl_alphabet_train/asl_alphabet_train',
                                         target_size=(150,150),
                                         class_mode='categorical',
                                         subset="training",
                                        shuffle='True'
                                       )
val_dataset=train.flow_from_directory('../input/asl-alphabet/asl_alphabet_train/asl_alphabet_train',
                                         target_size=(150,150),
                                         class_mode='categorical',
                                         subset="validation",
                                      shuffle='True'
                                       )
train_dataset.class_indices
model=Sequential()
model.add(Conv2D(64,(3,3),kernel_initializer='he_uniform',input_shape=(150,150,3),activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(64,(3,3),kernel_initializer='he_uniform',activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(128,(5,5),kernel_initializer='he_uniform',activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(128,(5,5),kernel_initializer='he_uniform',activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(256,(3,3),kernel_initializer='he_uniform',activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(256,(3,3),kernel_initializer='he_uniform',activation='relu',padding='same'))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256,activation='relu',kernel_initializer='he_uniform',kernel_regularizer='l2'))
model.add(Dense(128,activation='relu',kernel_initializer='he_uniform',kernel_regularizer='l2'))
model.add(Dense(64,activation='relu',kernel_initializer='he_uniform',kernel_regularizer='l2'))
model.add(Dense(29,activation='softmax'))
print(model.summary())
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
history=model.fit(train_dataset,epochs=1,validation_data=val_dataset)
def check(val):
    if(val<=25):
        print(chr(65+val))
    else:
        if(val==26):
            print('del')
        if(val==27):
            print('nothing')
        if(val==28):
            print('space')
    
img=image.load_img('../input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/M_test.jpg',target_size=(150,150))
img1=image.img_to_array(img)
img1=img1/255
img1=np.expand_dims(img1,[0])
check(np.argmax(model.predict(img1)))
