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
df_train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

df_test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
Y = df_train['label']

X = df_train.drop(columns = ['label'])
X[:1]
X[:1].shape
X = X.values.reshape(-1,28,28,1)
X.shape
from keras.utils.np_utils import to_categorical

Y = to_categorical(Y,num_classes = 10)
Y[0]
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

%matplotlib inline
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2,random_state = 0)
g = plt.imshow(X_train[98][:,:,0])
from keras.models import Sequential

from keras.layers import Dense,Activation,Dropout,Flatten,Conv2D,MaxPooling2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',

                 input_shape=X_train.shape[1:]))

model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3), padding='same'))

model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(10))

model.add(Activation('softmax'))

opt = RMSprop(learning_rate=0.0001, decay=1e-6)

model.compile(loss='categorical_crossentropy',

              optimizer=opt,

              metrics=['accuracy'])
model.fit(X_train, y_train,

              batch_size=25,

              epochs=50,

              validation_data=(X_test, y_test),

              shuffle=True)
Test = df_test.values.reshape(-1,28,28,1)
results = model.predict(Test)
results = np.argmax(results,axis = 1)
results.shape
results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_mnist.csv",index=False)