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
import numpy as np

from keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D

from keras.models import Sequential, save_model

from keras.utils import np_utils

from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle

import os

import cv2
path = '/kaggle/input/sign-language-gesture-images-dataset/Gesture Image Pre-Processed Data/'
gestures = os.listdir(path)
dict_labels = {

    '_':1,

    '0':2,

    '1':3,

    '2':4,

    '3':5,

    '4':6,

    '5':7,

    '6':8,

    '7':9,

    '8':10,

    '9':11,

    'A':12,

    'B':13,

    'C':14,

    'D':15,

    'E':16,

    'F':17,

    'G':18,

    'H':19,

    'I':20,

    'J':21,

    'K':22,

    'L':23,

    'M':24,

    'N':25,

    'O':26,

    'P':27,

    'Q':28,

    'R':29,

    'S':30,

    'T':31,

    'U':32,

    'V':33,

    'W':34,

    'X':35,

    'Y':36,

    'Z':37,

    

}
print(list(dict_labels.keys()))
x, y = [], []

for ix in gestures:

    images = os.listdir(path + ix)

    for cx in images:

        img_path = path + ix + '/' + cx

        img = cv2.imread(img_path, 0)

        img = img.reshape((50,50,1))

        img = img/255.0

        x.append(img)

        y.append(dict_labels[ix])
X = np.array(x)

Y = np.array(y)

Y = np_utils.to_categorical(Y)

X, Y = shuffle(X, Y, random_state=0)

categories = Y.shape[1]
X.shape
Y.shape
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
print(X_train.shape, X_test.shape)

print(Y_train.shape, Y_test.shape)
model = Sequential()

model.add(Conv2D(64, kernel_size=(3,3), activation = 'relu', input_shape=(50,50 ,1) ))

model.add(MaxPooling2D(pool_size = (2, 2)))



model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))



model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))



model.add(Flatten())

model.add(Dense(128, activation = 'relu'))

model.add(Dropout(0.20))

model.add(Dense(categories, activation = 'softmax'))



model.summary()
model.compile(optimizer='Adam', metrics=['accuracy'], loss='categorical_crossentropy')
fit = model.fit(X_train, Y_train, batch_size=138, epochs=5, validation_data=[X_test, Y_test])
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score
y_pred.round()
accuracy_score(Y_test, y_pred.round())*100
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_pred.round(), Y_test))
accuracy = model.evaluate(X_test,Y_test,batch_size=138)

print("Accuracy: ",accuracy[1]*100)
import matplotlib.pyplot as plt

%matplotlib inline
plt.plot(fit.history['acc'])

plt.plot(fit.history['val_acc'])

plt.title("Accuracy")

plt.xlabel('epoch')

plt.ylabel('accuracy')

plt.legend(['train','test'])

plt.show()
plt.plot(fit.history['loss'])

plt.plot(fit.history['val_loss'])

plt.title("Loss")

plt.xlabel('epoch')

plt.ylabel('loss')

plt.legend(['train','test'])

plt.show()