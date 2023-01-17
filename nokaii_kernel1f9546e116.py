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
import pandas as pd

import cv2

import os

import numpy as np

from keras import applications

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
prelink = '/kaggle/input/imagecitylandscape/'

def open_imgs(links):

    imgs = []

    for i in links:

        img = cv2.imread(prelink+i)

        imgs.append(img)

    return imgs
city = pd.read_csv(prelink+'city.csv')

lands = pd.read_csv(prelink+'landscape.csv')
data_340_510 = pd.concat([city, lands])

data_340_510['labelInt'] = data_340_510.label.apply(lambda x: 1 if x == 'city' else 0)
from keras.models import Sequential

from keras.layers import Dense, Conv2D, Flatten

#create model

model = Sequential()

#add model layers

model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(340,510,3)))

model.add(Conv2D(32, kernel_size=3, activation='relu'))

model.add(Flatten())

model.add(Dense(2, activation='softmax'))
#compile model using accuracy to measure model performance

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#train the model

dataX = open_imgs(data_340_510.img.values)

dataY = data_340_510['labelInt']

dataX = np.asarray(dataX)

dataY = np.asarray(dataY)



from keras.utils import to_categorical

dataY = to_categorical(dataY)



X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.2)
print(y_test.shape)

print('Num label 1:',(y_test == [0,1]).sum()/2)

print('Num label 0:',(y_test == [1,0]).sum()/2)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20)
base_model = applications.resnet50.ResNet50(weights= None, include_top=False, input_shape= (340,510,3))
from keras.models import Sequential,Model,load_model

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dropout(0.7)(x)

predictions = Dense(2, activation= 'softmax')(x)

model = Model(inputs = base_model.input, outputs = predictions)
from keras.optimizers import SGD, Adam

# sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)

adam = Adam(lr=0.0001)

model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs = 50, batch_size=10)