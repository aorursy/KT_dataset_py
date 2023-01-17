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
import pandas as pd

import numpy as np



import matplotlib.pyplot as plt



import seaborn as sns



import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D,BatchNormalization

from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop

from keras import backend as K



dataf = pd.read_csv('../input/train.csv')

dataf_test = pd.read_csv('../input/test.csv')



x_train = dataf.iloc[:,1:].values

y_train = dataf.iloc[:,0].values

x_test = dataf_test.iloc[:,:].values



batch_size = 100

num_classes = 10

epochs = 15



# input image dimensions

img_rows, img_cols = 28, 28



x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)

x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

input_shape = (img_rows, img_cols, 1)



x_train = x_train.astype('float32')

x_test = x_test.astype('float32')

x_train /= 255

x_test /= 255



y_train = keras.utils.to_categorical(y_train, num_classes)



model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])



model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs)



y_test = model.predict(x_test)



pred = np.argmax(y_test,axis=1)



img_id = []

for i in range(len(pred)):

  img_id.append(i+1)



d = {'ImageId':img_id,'Label':pred}

submission = pd.DataFrame(d)

submission.to_csv('./submission.csv',index = False)