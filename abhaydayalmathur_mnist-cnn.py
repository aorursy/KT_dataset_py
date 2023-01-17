# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Flatten, MaxPooling2D, Conv2D, BatchNormalization





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



print(tf.__version__)



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
submission = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")

submission.head()
traindat = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

traindat.head()
testdat = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

testdat.head()
Y = traindat['label']

X = traindat.drop(['label'], axis = 1)

X.head()
Y = np.array(Y)

X = np.array(X)/255.

X = np.array([x.reshape((28, 28, 1)) for x in X])



X = np.append(X, X, axis = 0)

Y = np.append(Y, Y, axis = 0)



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 40)



print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
model = Sequential([

  

    Conv2D(input_shape =(28,28, 1), filters = 128, kernel_size = (3, 3), activation = 'relu'),

    BatchNormalization(),

    MaxPooling2D(pool_size = (2,2)),

    Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu'),

    BatchNormalization(),

    Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu'),

    BatchNormalization(),

    MaxPooling2D(pool_size = (2,2)),

    Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'),

    BatchNormalization(),

    Flatten(),

    Dense(128, activation = 'relu'),

    BatchNormalization(),

    Dropout(0.5),

    Dense(64, activation = 'relu'),

    Dense(10, activation = 'softmax')

])



model.compile(optimizer = 'adam', metrics = ['accuracy'], loss = 'sparse_categorical_crossentropy')



model.summary()
history = model.fit(X_train, y_train,batch_size = 10000, epochs = 35, validation_data  = (X_test, y_test))
model.fit(X, Y, epochs = 10)
testdat = np.array(testdat)/255.

testdat = np.array([x.reshape(28, 28, 1) for x in testdat])

print(testdat.shape)
test_preds = model.predict(testdat)

print(test_preds.shape)

test_preds = np.array([np.argmax(y) for y in test_preds])

print(test_preds.shape)

testId = [*range(1, len(test_preds)+1)]
output = pd.DataFrame({'ImageId': testId,

                      'Label': test_preds})

output.to_csv('submission.csv', index=False)
