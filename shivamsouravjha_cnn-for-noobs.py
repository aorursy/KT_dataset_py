# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

import tensorflow as tf

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
train= pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test= pd.read_csv('/kaggle/input/digit-recognizer/test.csv')


train.tail()
Ytrain= train['label'].astype('float32')





train= train.drop('label',axis=1)


print(train.shape[0])
train= train.values.reshape(-1,28,28,1).astype('float32')

test= test.values.reshape(-1,28,28,1).astype('float32')



train =train / 255.0

test = test / 255.0

from keras.utils.np_utils import to_categorical

Ytrain=to_categorical(Ytrain,num_classes=10)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(train,Ytrain,test_size=0.25)
from keras.models import Sequential # to create a cnn model

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

model = Sequential()
model.add(Conv2D(32,(3,3),padding='same',activation= 'relu',input_shape=(28,28,1)))

model.add(BatchNormalization())

model.add(Conv2D(32,(3,3),padding='same',activation= 'relu',input_shape=(28,28,1)))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.2))
model.add(Conv2D(64,(3,3),padding='same',activation= 'relu'))

model.add(BatchNormalization())

model.add(Conv2D(64,(3,3),padding='same',activation= 'relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(256,activation= 'relu'))

model.add(Dropout(0.25))

model.add(Dense(128,activation= 'relu'))

model.add(Dropout(0.50))



model.add(Dense(10,activation= 'softmax'))
from keras.optimizers import RMSprop,Adam,SGD,Adagrad,Adadelta,Adamax,Nadam



optimizer =Adam(lr=0.004)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

model.summary()
from tensorflow.keras.utils import plot_model

plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

from IPython.display import Image

Image("model.png")

from keras.callbacks import LearningRateScheduler



annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)

datagen = ImageDataGenerator(zoom_range = 0.2,

                             )

datagen.fit(x_train)
hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size=256),

                           steps_per_epoch=600,

                           epochs=15, #Increase this when not on Kaggle kernel

                           verbose=1,  #1 for ETA, 0 for silent

                           validation_data=(x_test, y_test) #For speed

                            )
y_pred = model.predict(test, verbose = 1)

predictions=[]

for i in range(len(test)):

    a=np.where(y_pred[i] == max(y_pred[i]))

    predictions.append(a[0][0])
import pandas as pd

counter = range(1, len(predictions) + 1)

solution = pd.DataFrame({"ImageId": counter, "label": list(predictions)})

solution.to_csv("digit_recognizer8.csv", index = False)