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

#importing the model and layers

from keras.models import Sequential

from keras.layers import Convolution2D

from keras.layers import MaxPooling2D

from keras.layers import Dense

from keras.layers import Flatten



model=Sequential()



#adding the layers



model.add(Convolution2D(32,(3,3), input_shape=(64,64,3),activation='relu'))



model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Convolution2D(32,(3,3),activation='relu'))



model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Convolution2D(32,(3,3),activation='relu'))



model.add(Flatten())



model.add(Dense(32,input_dim=64,kernel_initializer='uniform',activation='relu'))



model.add(Dense(1,input_dim=1,activation='sigmoid',kernel_initializer='uniform'))



model.summary()



#image preprocessing

from keras.preprocessing.image import ImageDataGenerator

train_data = ImageDataGenerator(rescale=1./255,shear_range=0.2,horizontal_flip=True)

test_data = ImageDataGenerator(rescale=1./255)

#training the data

x_train=train_data.flow_from_directory('../input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train')

x_test=train_data.flow_from_directory('../input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test')

#compiling the dataset

model.compile(loss='binary_crossentropy',optimizer="adam",metrics=["accuracy"])



#fitting the model

model.fit_generator(x_train,steps_per_epoch=163,epochs=10,validation_data=x_test,validation_steps=20)



model.save("pcnn.h5")