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
import numpy as np

import pandas as pd
df_train = pd.read_csv("../input/fashion-mnist_train.csv")

df_train.shape
# Load test data 

df_test = pd.read_csv("../input/fashion-mnist_test.csv")

df_test.shape
# Global variables

IMAGE_HEIGHT= 28

IMAGE_WIDTH= 28

IMAGE_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
#Train data

X = np.array(df_train.iloc[:,1:])

y = to_categorical(np.array(df_train.iloc[:,0]),10)
X
#Test data

X_test = np.array(df_test.iloc[:,1:])

y_test = to_categorical(np.array(df_test.iloc[:,0]),10)
X_test
X.shape[0]
X = X.reshape(X.shape[0], IMAGE_HEIGHT, IMAGE_WIDTH,1)

X_test = X_test.reshape(X_test.shape[0], IMAGE_HEIGHT, IMAGE_WIDTH,1)
X = X.astype('float32')

X_test = X_test.astype('float32')

X = X/255

X_test = X_test/255
X.shape
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2,

                                                  random_state=12)
X_train.shape
X_val.shape
## reshape the inputs

#X_train = X_train.reshape(-1, 784)

#X_val = X_val.reshape(-1, 784)
from keras.models import Sequential

from keras.layers import Dense, Conv2D, Convolution2D, MaxPooling2D, Dropout, Flatten

from keras import backend as K

K.set_image_dim_ordering('tf') # MOST IMPORTANT

#input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 1)
model = Sequential()

model.add(Conv2D(32,(3,3), activation='relu',input_shape=(28,28,1)))

model.add(MaxPooling2D(pool_size = (2,2)))

mocel.add(Dropout(0.25))



model.add(Conv2D(64,(3,3), activation='relu'))

model.add(MaxPooling2D(pool_size= (2,2)))

mocel.add(Dropout(0.25))



model.add(Conv2D(128,(3,3), activation='relu'))

model.add(MaxPooling2D(pool_size= (2,2)))

mocel.add(Dropout(0.25))



model.add(Convolution2D(256,(3,3), activation='relu',padding='SAME'))

model.add(MaxPooling2D(pool_size= (2,2)))

mocel.add(Dropout(0.25))



model.add(Flatten())



model.add(Dense(128, activation='relu'))

mocel.add(Dropout(0.25))



model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
model.summary()
history = model.fit(X_train, y_train,

          batch_size=256,

          epochs=50,

          verbose=1,

          validation_data=(X_val, y_val))
score = model.evaluate(X_val,y_val)
score[0]
score[1]
predicted = model.predict(X_test)
labels = np.argmax(predicted, axis=-1)
len(labels)
print(labels)