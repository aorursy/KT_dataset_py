# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

test = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')
print('Training set:', train.shape)

print('Test set: ', test.shape)
X_train = train

X_test = test

Y_train = X_train.pop('label')

X_test.pop('id')
'''Preprocess'''

import numpy as np

X_train, X_test = np.array(X_train)/255.0, np.array(X_test)/255.0

X_train = X_train.reshape(X_train.shape[0], 28, 28,1)

X_test = X_test.reshape(X_test.shape[0], 28,28,1)
from sklearn.utils import shuffle

X_train, Y_train = shuffle(X_train, Y_train)
#Create validation set

from sklearn.model_selection import train_test_split

X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.2)

print('Training set:', X_train.shape, Y_train.shape)

print('Validation set:', X_valid.shape, Y_valid.shape)
from tensorflow.keras.models import Model

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D

from tensorflow.keras.layers import MaxPooling2D

from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=[3,3],input_shape=[28,28,1], padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=32, kernel_size=[3,3], padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dense(10, activation='softmax'))
model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=1024, epochs=20, validation_data=(X_valid, Y_valid))
test.head(5)
test = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')

id = test['id']
predict = model.predict(X_test)
predict = np.argmax(predict, axis=1)
predict.shape
submission = pd.DataFrame({'id':id,'label':predict})
filename = 'submission.csv'

submission.to_csv(filename, index=False)
submission.head(5)