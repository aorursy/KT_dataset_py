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
dataset=pd.read_csv('../input/train.csv')
y=dataset['label']
X_train=dataset.iloc[:,1:]
X_test=pd.read_csv('../input/test.csv')
X_train=X_train/255.0

X_test=X_test/255.0
X_train=X_train.values.reshape(-1,28,28,1)
X_test=X_test.values.reshape(-1,28,28,1)
from keras.models import Sequential

from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D

from keras.preprocessing.image import ImageDataGenerator

from keras.utils.np_utils import to_categorical

from sklearn.model_selection import train_test_split, cross_val_score

import itertools
target=to_categorical(y,10)
X_t, X_v, Y_t, Y_v = train_test_split(X_train, target, test_size = 0.1)
model=Sequential()

model.add(Conv2D(filters=64,kernel_size=(7,7),padding = 'Same', 

                     activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters=64,kernel_size=(7,7),padding = 'Same', 

                     activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.3))
model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', 

                     activation ='relu'))

model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', 

                     activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.3))



model.add(Flatten())

model.add(Dense(512, activation = "relu", use_bias= True))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))
model.compile(optimizer = 'adam' , loss = "categorical_crossentropy", metrics=["accuracy"])
datagen = ImageDataGenerator(

        rotation_range=10,  

        zoom_range = 0.2, 

        width_shift_range=0.2,  

        height_shift_range=0.1,  

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)

datagen.fit(X_t)
model.fit_generator(datagen.flow(X_t,Y_t, batch_size= 82),

                              epochs = 60, validation_data = (X_v,Y_v),

                              verbose = 2, steps_per_epoch=X_t.shape[0] // 82)

ewsult=model.predict(X_test)
ewsult=np.argmax(ewsult,axis=1)
ewsult = pd.Series(ewsult,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),ewsult],axis = 1)



submission.to_csv("cnnmodel.csv",index=False)