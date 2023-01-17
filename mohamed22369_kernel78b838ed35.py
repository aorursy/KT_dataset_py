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

from sklearn.model_selection import train_test_split

import itertools

from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D

from keras.optimizers import RMSprop

from keras.callbacks import ReduceLROnPlateau
Y_train=pd.read_csv("/kaggle/input/ahdd1/csvTrainLabel 60k x 1.csv")

X_train=pd.read_csv("/kaggle/input/ahdd1/csvTrainImages 60k x 784.csv")

test=pd.read_csv("/kaggle/input/ahdd1/csvTestImages 10k x 784.csv")
Y_train=Y_train.iloc[:,0]

X_train=X_train/255.0

test=test/255.0

X_train=X_train.values.reshape(-1, 28,28,1)

test=test.values.reshape(-1, 28, 28, 1)

Y_train=to_categorical(Y_train, num_classes=10)

random_seed=2

X_train,X_val,Y_train,Y_val=train_test_split(X_train, Y_train, test_size=0.1, random_state=random_seed)
img_rows = X_train[0].shape[0]

img_cols = X_train[1].shape[0]

input_shape = (img_rows, img_cols, 1)

batch_size = 128

epochs = 10

num_classes = Y_val.shape[1]

num_pixels = X_train.shape[1] * X_train.shape[2]



from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras import backend as K

from keras.optimizers import SGD 





# create model

model = Sequential()



model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=input_shape))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss = 'categorical_crossentropy',

              optimizer = SGD(0.01),

              metrics = ['accuracy'])



print(model.summary())



history = model.fit(X_train, Y_train,

          batch_size=batch_size,

          epochs=epochs,

          verbose=1,

          validation_data=(X_val, Y_val))



score = model.evaluate(X_val, Y_val, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])