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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline


train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
Y_train = train["label"]



# Drop 'label' column

X_train = train.drop(labels = ["label"],axis = 1) 



# free some space

del train 



g = sns.countplot(Y_train)



Y_train.value_counts()
X_train = X_train/255

test = test/255
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)



Y_train = to_categorical(Y_train, num_classes = 10)
from sklearn.model_selection import train_test_split

random_seed=2

X_train, X_Val, Y_train, Y_Val = train_test_split(X_train, Y_train, test_size = 0.2, random_state=random_seed)
g = plt.imshow(X_train[0][:,:,0])
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout,GlobalAveragePooling2D, MaxPool2D



model = Sequential()

model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu', 

                        input_shape=(28,28,1)))

model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.2))

model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.2))

model.add(Conv2D(filters=128, kernel_size=2, padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.2))



model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.4))

model.add(Dense(10, activation = "softmax"))



model.summary()


model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
from keras.callbacks import ModelCheckpoint

epochs = 10





#checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', 

                               #verbose=1, save_best_only=True)

model.fit(X_train, Y_train, 

          validation_data=(X_Val, Y_Val),

          epochs=epochs, batch_size=20, verbose=1)


score = model.evaluate(X_Val, Y_Val, verbose=0)

print('\n', 'Test accuracy:', score[1])
image_index = 1000

plt.imshow(X_Val[image_index].reshape(28, 28),cmap='Greys')

pred = model.predict(X_Val[image_index].reshape(-1,28,28,1))

print(pred.argmax())