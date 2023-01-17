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
import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D

from sklearn.model_selection import train_test_split



np.random.seed(42)
sub = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

X = train.drop('label',axis=1)
X_norm = X/255

test_norm = test/255
y = train['label']
y_dum = pd.get_dummies(y)
test_resize = test/255
X_norm.max().max()
X_norm
X_norm.values[0,:].reshape(28,28)
X_reshape = []

for i in X_norm.values:

    X_reshape.append(i.reshape(28,28))



X_rsarr = np.array(X_reshape)
X_2d = X_rsarr.reshape(-1, 28,28,1)
X_2d.shape
test_reshape = []

for i in test_norm.values:

    test_reshape.append(i.reshape(28,28))

test_rsarr = np.array(test_reshape)

test_2d = test_rsarr.reshape(-1, 28,28,1)
X_train, X_test, y_train, y_test = train_test_split(X_2d, y_dum)
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=(28,28,1)))

model.add(Conv2D(32, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128))

model.add(Activation('sigmoid'))

model.add(Dense(32))

model.add(Activation('sigmoid'))

model.add(Dropout(0.5))

model.add(Dense(10))

model.add(Activation('sigmoid'))

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
model.fit(X_train, y_train,

          batch_size=32,

          epochs=10,

          validation_data=(X_test, y_test), 

          verbose=2)
model.fit(X_2d, y_dum,

          batch_size=32,

          epochs=5, 

          verbose=2)
sub
y_pred = model.predict(test_2d)
y_pred


y_max = np.argmax(y_pred,axis = 1)





y_max


sub['Label'] = y_max
sub
sub.to_csv('submission.csv', index=False)