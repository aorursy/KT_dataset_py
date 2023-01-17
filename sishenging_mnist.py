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
import keras

from keras.models import Sequential

from keras.layers import Dense , Flatten , Conv2D , MaxPooling2D

from keras import  backend as K
train = pd.read_csv("../input/train.csv")

print(train.shape)

train.head()
test= pd.read_csv("../input/test.csv")

print(test.shape)

test.head()
X_train = (train.iloc[:,1:].values).astype('float32') # all pixel values

y_train = train.iloc[:,0].values.astype('int32') # only labels i.e targets digits

X_test = test.values.astype('float32')
X_train = X_train.reshape(X_train.shape[0], 28, 28,1)

X_test = X_test.reshape(X_test.shape[0], 28, 28,1)

X_train /= 255

X_test /= 255

X_train.shape
y_train= keras.utils.to_categorical(y_train,10)
model = Sequential()

model.add(Conv2D(32,kernel_size=(5,5),activation='relu',input_shape=(28,28,1)))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,kernel_size=(5,5),activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(500,activation='relu'))

model.add(Dense(10,activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,

             optimizer=keras.optimizers.SGD(),

             metrics=['accuracy'])
model.fit(X_train, y_train,

         batch_size=128,

         epochs=20)
predictions = model.predict_classes(X_test, verbose=0)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": predictions})

submissions.to_csv("DR.csv", index=False, header=True)