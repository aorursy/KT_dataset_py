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
import tensorflow

from tensorflow import keras
%matplotlib inline
import matplotlib.pyplot as plt
print('Loading data...')

train = pd.read_csv('../input/digit-recognizer/train.csv')

test_X = pd.read_csv('../input/digit-recognizer/test.csv')
print(train.shape)

print(test_X.shape)
train.head()
test_X.head()
train_X = train.iloc[:, 1:].values.astype('float32')

train_y = train.iloc[:, 0].values.astype('float32') 
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = map(lambda x : np.array(x).astype(np.float32), train_test_split(train_X, train_y, test_size=0.2))
X_train.shape
for i in range(10):

    plt.subplot(2, 5, i+1)

    plt.title("label:" + str(i+100))

    plt.imshow(X_train[i+100].reshape(28, 28), cmap=None)

    

#Display an image.
y_train[100:110]
X_train, X_val = X_train / 255.0, X_val / 255.0
model = keras.models.Sequential([

    keras.layers.Flatten(),

    keras.layers.Dense(512, activation='relu'),

    keras.layers.Dropout(0.2),

    keras.layers.Dense(10, activation='softmax')

])
model.compile(optimizer='adam',

             loss='sparse_categorical_crossentropy',

             metrics=['accuracy'])
history = model.fit(X_train, y_train,

                    validation_data=(X_val, y_val), epochs=10,

                    batch_size=100, verbose=1)
model.evaluate(X_val, y_val)
y_val[0:10]
for i in range(10):

    plt.subplot(2, 5, i+1)

    plt.title("label:" + str(i))

    plt.imshow(X_val[i].reshape(28, 28), cmap=None)

    

#Display an image.
model.predict(X_val[0:10])
y_pred = model.predict(test_X, batch_size=2000, 

                       verbose=1)
y_pred = np.argmax(y_pred, axis=1)

to_submit = pd.DataFrame({'Label': y_pred})
to_submit.index += 1

to_submit.index.name = "ImageId"

to_submit.to_csv('submition_3layer_keras.csv')