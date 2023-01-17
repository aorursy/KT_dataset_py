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
train_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')



print (train_data.shape)

print (test_data.shape)
X = train_data.iloc[:, 1:].values

y = train_data.iloc[:, 0].values



print (X.shape)

print (y.shape)
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

X = scaler.fit_transform(X)
X_copy = X.copy()

X_copy.shape
X = X_copy.copy()

X = X.reshape(X.shape[0], 28, 28, 1)

X.shape
y = y.reshape(y.shape[0], 1)

print (y.shape)
from sklearn.preprocessing import OneHotEncoder



enc = OneHotEncoder(sparse=False)

y = enc.fit_transform(y)



print (y.shape)
import tensorflow as tf



model = tf.keras.models.Sequential()



model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[28, 28, 1]))

model.add(tf.keras.layers.MaxPool2D())



model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))

model.add(tf.keras.layers.MaxPool2D())



model.add(tf.keras.layers.Flatten())



# model.add(tf.keras.layers.Dense(units=784, activation='relu'))

model.add(tf.keras.layers.Dense(units=50, activation='relu'))

model.add(tf.keras.layers.Dense(units=50, activation='relu'))

model.add(tf.keras.layers.Dense(units=10, activation='softmax'))



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, batch_size=32, epochs=100)
y_hat = model.predict(X)

print (y_hat.shape)
y_hat = enc.inverse_transform(y_hat)

print (y_hat.shape)
y_hat = y_hat.reshape(y_hat.shape[0],)

print (y_hat.shape)
y = enc.inverse_transform(y)

print (y.shape)
y = y.reshape(y.shape[0],)

print (y.shape)
from sklearn.metrics import confusion_matrix



confusion_matrix(y, y_hat)
test_data.shape
X_test = test_data.iloc[:, :].values

print (X_test.shape)
X_test = scaler.transform(X_test)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_test.shape
y_pred = model.predict(X_test)

print (y_pred.shape)
y_pred = enc.inverse_transform(y_pred)

print (y_pred.shape)
y_pred = y_pred.reshape(y_pred.shape[0],)

print (y_pred.shape)
submission_data = pd.DataFrame()

submission_data['ImageId'] = test_data.index + 1

submission_data['Label'] = y_pred

submission_data.head()
submission_data.to_csv('submission.csv', index=False)