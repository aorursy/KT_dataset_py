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
route = '../input/digit-recognizer/'

train = pd.read_csv(f'{route}train.csv')

test = pd.read_csv(f'{route}test.csv')

print(train.shape)

print(test.shape)
label = train.label

train = train.drop('label', axis=1)

train.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train, label, random_state=0)
#labelのone-hot-encoder

from keras.utils import to_categorical

y_train = to_categorical(y_train, 10)

y_test = to_categorical(y_test, 10)
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
from keras.layers import Dense, Dropout

from keras.models import Sequential

from keras.metrics import accuracy



model = Sequential()

model.add(Dense(64, activation='relu', input_shape=(784,)))

model.add(Dropout(0.2))

model.add(Dense(32, activation='relu'))

model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
# モデルの学習

history = model.fit(X_train, y_train, epochs=30, batch_size=128, verbose=1,

                   validation_data=(X_test, y_test))
import matplotlib.pyplot as plt

loss = history.history['loss']

val_loss = history.history['val_loss']



num_epoch = len(loss)

plt.plot(range(num_epoch), loss, marker='.', label='loss')

plt.plot(range(num_epoch), val_loss, marker='.', label='val_loss')

plt.legend(loc='best')

plt.grid()

plt.xlabel('epoch')

plt.ylabel('loss')

plt.show()
accuracy = history.history['accuracy']

val_accuracy = history.history['val_accuracy']



num_epoch = len(accuracy)

plt.plot(range(num_epoch), accuracy, marker='.', label='accuracy')

plt.plot(range(num_epoch), val_accuracy, marker='.', label='val_accuracy')

plt.legend(loc='best')

plt.grid()

plt.xlabel('epoch')

plt.ylabel('accuracy')

plt.show()
test = pd.read_csv(f'{route}test.csv')
test.head()
test_pred = model.predict_classes(test)
submission = pd.read_csv(f'{route}sample_submission.csv')

submission.head()
submission['Label'] = test_pred

submission.head()
submission.to_csv('ANSWER.csv',header=True, index=False)