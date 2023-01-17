# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn.model_selection import train_test_split



from keras.utils.np_utils import to_categorical

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from tensorflow.keras import utils
train = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')
X_train = train.drop(labels = ['label'], axis=1)

y_train = train['label']



X_train = X_train / 255.0

test = test / 255.0



X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)



y_train = to_categorical(y_train, num_classes = 10)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 

                                                  test_size = 0.1,

                                                  random_state=42)
model = Sequential()

model.add(Conv2D(filters=16,

                 kernel_size=(3,3),

                 activation='relu',

                 input_shape=(28,28,1)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64,

                 kernel_size=(3, 3),

                 activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(32, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
history = model.fit(X_train,

                   y_train,

                   batch_size=256,

                   validation_data=(X_val,y_val),

                   epochs=10,

                   verbose=1)
train_loss = history.history['loss']

test_loss = history.history['val_loss']



plt.figure(figsize=(12,8))



plt.plot(train_loss, label='Training Loss')

plt.plot(test_loss, label='Validation Loss')



plt.title('Training and Validation Loss by Epoch', fontsize = 25)

plt.xlabel('Epoch', fontsize = 18)

plt.ylabel('Categorical Crossentropy', fontsize = 18)

plt.xticks(np.arange(10), np.arange(10))



plt.legend(fontsize = 10);
res = model.predict(test)



res = np.argmax(res, axis=1)

res = pd.Series(res, name='label')



res.shape
sub = pd.concat([pd.Series(range(1,28001), name = 'ImageId'), res], axis=1)



sub.to_csv('mnist-cnn.csv',index=False)