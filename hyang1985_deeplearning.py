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
from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation, Flatten

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.optimizers import SGD, Adam, RMSprop

from keras.utils import np_utils

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
BATCH_SIZE = 64

NB_EPOCH = 20

NB_CLASSES = 10

VERBOSE = 1

VALIDATION_SPLIT = 0.2

OPTIM = RMSprop()
train=pd.read_csv("../input/digit-recognizer/train.csv")

test=pd.read_csv("../input/digit-recognizer/test.csv")

train_Y=train['label']

train_X=train.loc[:,'pixel0':'pixel783']

#train_Y,test_Y=train['label'][:40000],train['label'][40000:]

#train_X,test_X=train.loc[:40000,'pixel0':'pixel783'],train.loc[40000:,'pixel0':'pixel783']

print("original shape of train is %d and %d"%(train_X.shape[0],train_X.shape[1]))
train_X=train_X/255
train_X=np.array(train_X).reshape(42000,28,28,1)

X_train=train_X[:40000]

X_test=train_X[40000:]



Y_train=train_Y[:40000]

Y_test=train_Y[40000:]

Y_train = np_utils.to_categorical(Y_train,10)

Y_test = np_utils.to_categorical(Y_test,10)
X_train.shape
Y_train.shape
Y_test.shape
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

model.add(Dense(NB_CLASSES))

model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer=OPTIM,

metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=BATCH_SIZE,

epochs=NB_EPOCH, validation_split=VALIDATION_SPLIT,

verbose=VERBOSE)

score = model.evaluate(X_test, Y_test,

batch_size=BATCH_SIZE, verbose=VERBOSE)

print("Test score：", score[0])

print('Test accuracy：', score[1])

test=np.array(test).reshape(test.shape[0],28,28,1)

test=test/255
y_test=model.predict_classes(test)
ids=range(1,len(y_test)+1)

submission=pd.DataFrame(columns=['ImageId','Label'])
submission['ImageId']=ids

submission['Label']=y_test
submission.to_csv("submission.csv", index=False)