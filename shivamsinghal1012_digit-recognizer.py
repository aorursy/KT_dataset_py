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
import matplotlib.pyplot as plt

from keras import utils

from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation, Flatten 

from keras.utils import to_categorical

from keras.layers import Conv2D, MaxPooling2D
train  = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')

train.head()
Y_train = train['label']

X_train = train.drop(labels = ['label'],axis = 1)
X_train = np.array(X_train)

test = np.array(test)

X_train = X_train.reshape(-1,28,28,1)

test = test.reshape(-1,28,28,1)
Y_train = to_categorical(Y_train, 10)
print(X_train.shape)

print(test.shape)

print(Y_train.shape)
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy',

              optimizer= "adam",

              metrics=['accuracy'])



history = model.fit(X_train, Y_train,

                    batch_size= 128,

                    epochs= 20, shuffle = True,

                    validation_split = 0.25)
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'])



plt.show()



plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'])

plt.show()
print(history.history)
results = model.predict(test)

results = np.argmax(results,axis = 1)

results = pd.Series(results,name = "Label")



submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("submission_file.csv",index = False)