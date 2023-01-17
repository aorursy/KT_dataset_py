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
test_path = '../input/test/'

train_path = '../input/train/train/'

train_df = pd.read_csv('../input/train.csv')
train_df.has_cactus.value_counts()
import matplotlib.pyplot as plt

import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split as tts

x_train,x_test=tts(train_df,test_size=0.2)
from keras.preprocessing.image import ImageDataGenerator
image_gen = ImageDataGenerator(shear_range=0.01,

                               zoom_range=[0.9, 1.25],

                               rescale=1./255,                               

                               horizontal_flip=True,

                               vertical_flip=True,

                               fill_mode='reflect',

                               brightness_range=[0.5, 1.5])

x_train.has_cactus=x_train.has_cactus.astype(str)

x_test.has_cactus=x_test.has_cactus.astype(str)

train_gen= image_gen.flow_from_dataframe(x_train,

                                        directory=train_path,

                                         target_size=(32,32),

                                         x_col='id',

                                         y_col='has_cactus',

                                         batch_size=64)

test_gen= image_gen.flow_from_dataframe(x_test,

                                        directory=train_path,

                                        target_size=(32,32),

                                        x_col='id',

                                        y_col='has_cactus',

                                        batch_size=64)



                                    
from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight('balanced', np.unique(train_gen.classes), train_gen.classes)

class_weights
from keras import Sequential

from keras.layers.convolutional import Conv2D, AveragePooling2D

from keras.optimizers import Adam

from keras.layers import Dense, Flatten,InputLayer,Input

from keras import backend as K

from keras import layers

from keras import utils as u

amodel=Sequential()

#amodel.add(InputLayer((1,32,32,3)))

amodel.add(Conv2D(6,(5,5),strides=1,padding='same',activation='tanh',input_shape=(32,32,3)))

amodel.add(AveragePooling2D((2,2),strides=2,padding='same'))

amodel.add(Conv2D(16,(5,5),strides=1,padding='same',activation='tanh'))

amodel.add(AveragePooling2D((2,2),strides=2))

amodel.add(Flatten())

amodel.add(Dense(120,activation='tanh'))

amodel.add(Dense(84,activation='tanh'))

amodel.add(Dense(2,activation='softmax'))



amodel.compile(loss='binary_crossentropy',

             optimizer='adam',

             metrics=['accuracy'])



amodel.summary()
amodel.fit_generator(train_gen,class_weight=class_weights,validation_data=test_gen,validation_steps=len(x_test)//64,steps_per_epoch=(len(x_train)//64),epochs=15)
eval_generator=ImageDataGenerator(rescale=1./255)

eval_gen= eval_generator.flow_from_directory(

                            directory=test_path,

                            target_size=(32,32),

                            class_mode=None,

                            batch_size=1,

                            shuffle=False)
submission= pd.read_csv('../input/sample_submission.csv')

file_name= [path.split('/')[-1] for path in eval_gen.filenames]

prob=list(amodel.predict_generator(eval_gen,steps=len(eval_gen))[:,0])





submission.id=file_name

submission.has_cactus=prob



submission.to_csv('sample_submission.csv',index=False)


