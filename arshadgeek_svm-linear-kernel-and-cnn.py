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
train = pd.read_csv("../input/sign-language-mnist/sign_mnist_train.csv")

test  = pd.read_csv("../input/sign-language-mnist/sign_mnist_test/sign_mnist_test.csv")
test.head(10)
X_train = train.drop(['label'], axis = 1)

y_train = train['label']



X_test  = test.drop(['label'], axis = 1)

y_test  = test['label']
X_train = X_train / 255

X_test  = X_test  /255
from sklearn.svm import SVC



svclassifier = SVC(kernel='linear')



svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix



print(classification_report(y_test,y_pred))
import keras 

from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten

from keras.utils import to_categorical 

import matplotlib.pyplot as plt

from keras.models import Sequential 
y_test = to_categorical(y_test)
X_train = np.array(X_train.iloc[:,:])

X_train = np.array([np.reshape(i, (28,28)) for i in X_train])



X_test = np.array(X_test.iloc[:,:])

X_test = np.array([np.reshape(i, (28,28)) for i in X_test])
num_classes = 26

y_train = np.array(y_train).reshape(-1)

y_test = np.array(y_test).reshape(-1)



y_train = np.eye(num_classes)[y_train]

y_test = np.eye(num_classes)[y_test]
X_train = X_train.reshape((27455, 28, 28, 1))

X_test = X_test.reshape((7172, 28, 28, 1))
classifier = Sequential()

classifier.add(Conv2D(filters=8, kernel_size=(3,3),strides=(1,1),padding='same',input_shape=(28,28,1),activation='relu', data_format='channels_last'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Conv2D(filters=16, kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'))

classifier.add(Dropout(0.5))

classifier.add(MaxPooling2D(pool_size=(4,4)))

classifier.add(Dense(128, activation='relu'))

classifier.add(Flatten())

classifier.add(Dense(26, activation='softmax'))

classifier.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])

classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

classifier.fit(X_train, y_train, epochs=50, batch_size=100)
accuracy = classifier.evaluate(x=X_test,y=y_test,batch_size=32)

print("Accuracy: ",accuracy[1])