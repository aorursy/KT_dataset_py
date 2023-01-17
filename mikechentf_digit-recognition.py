# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import matplotlib

import matplotlib.pyplot as plt



import ipywidgets as widgets

from ipywidgets import interact, interact_manual



from sklearn.model_selection import train_test_split



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
train.head()
X_train, X_val, y_train, y_val = train_test_split(train.iloc[:, 1:], train.iloc[:, 0], test_size=0.2)

print('X_train:', X_train.shape)

print('y_train:', y_train.shape)

print('X_val:', X_val.shape)

print('y_val:', y_val.shape)

@interact

def show_digital(x=(0, 1000)):

    return plt.imshow(np.array(X_train.iloc[x]).reshape((28,28))), print('The number is', y_train.iloc[x])

def normalization(X):

    X = X / 255.0

    return X
X_train = np.array(normalization(X_train))

X_val = np.array(normalization(X_val))

y_train = np.array(y_train)

y_val = np.array(y_val)
%%time

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression(solver='lbfgs', multi_class='multinomial').fit(X_train, y_train)

print('Score:', LR.score(X_val, y_val))
%%time

from sklearn.svm import LinearSVC

SVM = LinearSVC().fit(X_train, y_train)

print('Score:', SVM.score(X_val, y_val))
%%time

from sklearn.neighbors import KNeighborsClassifier

KNN = KNeighborsClassifier(n_neighbors=5, n_jobs=-1).fit(X_train, y_train)

print('Score:', KNN.score(X_val, y_val))
from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.utils import to_categorical

from keras.losses import categorical_crossentropy

from keras.optimizers import Adadelta
X_train = X_train.reshape(-1, 28, 28)

X_train = np.expand_dims(X_train, axis=1)

X_val = X_val.reshape(-1, 28, 28)

X_val = np.expand_dims(X_val, axis=1)

y_train = to_categorical(y_train, 10)

y_val = to_categorical(y_val, 10)

X_train.shape
def CNN_model():

    model = Sequential()

    model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(1,28,28), data_format='channels_first'))

    model.add(MaxPooling2D((2,2)))

    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))

    model.add(MaxPooling2D((2,2)))

    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))

    model.add(MaxPooling2D((2,2)))

    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    model.add(Dropout(0.2))

    model.add(Dense(128, activation='relu'))

    model.add(Dropout(0.2))

    model.add(Dense(128, activation='relu'))

    model.add(Dropout(0.2))

    model.add(Dense(10, activation='softmax'))

    return model
%%time

CNN = CNN_model()



CNN.compile(loss=categorical_crossentropy,

              optimizer=Adadelta(),

              metrics=['accuracy'])



CNN.fit(X_train, y_train,

          batch_size=128,

          epochs=30,

          verbose=1,

          validation_data=(X_val, y_val))

X_train = train.iloc[:, 1:]

y_train = train.iloc[:, 0]

X_test = test
X_train = np.array(normalization(X_train))

y_train = np.array(y_train)

X_test = np.array(normalization(X_test))
X_train = X_train.reshape(-1, 28, 28)

X_train = np.expand_dims(X_train, axis=1)

X_test = X_test.reshape(-1, 28, 28)

X_test = np.expand_dims(X_test, axis=1)

y_train = to_categorical(y_train, 10)
CNN = CNN_model()



CNN.compile(loss=categorical_crossentropy,

              optimizer=Adadelta(),

              metrics=['accuracy'])



CNN.fit(X_train, y_train,

          batch_size=128,

          epochs=30,

          verbose=1)
prediction_prob = CNN.predict(X_test)
prediction = np.argmax(prediction_prob, axis=1)
prediction
prediction_df = {"ImageId":range(1, X_test.shape[0]+1), "Label":prediction}

prediction_df = pd.DataFrame(prediction_df)

prediction_df.to_csv("prediction.csv", index = False)