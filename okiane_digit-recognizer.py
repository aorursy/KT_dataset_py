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
train = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')
train.head()
train.describe()
test.describe()
#Extract relevant columns

x_train = train.iloc[:,1:]

y_train = train['label']



x_test = test
# Convert labels to categorical data

from keras.utils import to_categorical

y_train = to_categorical(y_train)
import keras

from keras.callbacks import EarlyStopping

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
model = keras.Sequential() #Define sequential model

model.add(keras.layers.Dense(512, activation = 'relu', input_shape = (784,)))

model.add(keras.layers.Dropout(0.01))

model.add(keras.layers.Dense(256, activation = 'relu'))

model.add(keras.layers.Dropout(0.01))

model.add(keras.layers.Dense(128, activation = 'relu'))

model.add(keras.layers.Dropout(0.01))

model.add(keras.layers.Dense(64, activation = 'relu'))

model.add(keras.layers.Dropout(0.01))

model.add(keras.layers.Dense(32, activation = 'relu'))

model.add(keras.layers.Dropout(0.01))

model.add(keras.layers.Dense(16, activation = 'relu'))

model.add(keras.layers.Dropout(0.01))

model.add(keras.layers.Dense(10, activation = 'softmax'))

model.compile(optimizer = keras.optimizers.SGD(lr = 0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])

early_stopping_monitor = EarlyStopping(patience = 3) #Stop training after 3 epochs when there is no improvement in validation score



history = model.fit(x_train, y_train, epochs = 20, batch_size = 16, validation_split = 0.1, callbacks = [early_stopping_monitor])
import matplotlib.pyplot as plt



plt.plot(history.history['val_loss'])

plt.xlabel('Epochs')

plt.ylabel('Validation score')

plt.show()



plt.plot(history.history['accuracy'])

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.show()
from keras.models import load_model

model.save('model.h5')

model.summary()
#y_pred = model.predict(x_test)

y_pred = model.predict_classes(x_test)
pd.DataFrame(y_pred).describe()
#prob = y_pred[:, 1]
subm = pd.read_csv('../input/digit-recognizer/sample_submission.csv')

subm.set_index('ImageId')
subm['Label'] = y_pred
subm.set_index('ImageId')
subm.to_csv("submission.csv")