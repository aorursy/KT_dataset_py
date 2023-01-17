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
train = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')
X_train = train.drop("label",axis=1)

X_train = X_train.values.reshape(-1,28,28)

X_test = test.values.reshape(-1,28,28)

print(X_train.shape)

print(X_test.shape)
import tensorflow as tf

X_train = X_train / 255.0

X_test = X_test / 255.0

X_train = tf.expand_dims(X_train, 3)

X_test = tf.expand_dims(X_test, 3)
print(X_train.shape)

print(X_test.shape)
y_train = train.pop("label")

y_train = y_train.values

print(y_train.shape)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D,AveragePooling2D



# Initialize the model

model = Sequential()



# Add a Convolutional Layer

model.add(Conv2D(6, kernel_size=5, strides=1,  activation='tanh', input_shape=X_train[0].shape, padding='same'))



# Add a AvgPooling Layer

model.add(AveragePooling2D(pool_size=(2, 2)))



# Add a Convolutional Layer

model.add(Conv2D(16, kernel_size=5, strides=1,  activation='tanh', padding='valid'))



# Add a AvgPooling Layer

model.add(AveragePooling2D(pool_size=(2, 2)))



# Flatten the layer

model.add(Flatten())



# Add Fully Connected Layer with 120 units

model.add(Dense(120, activation="tanh"))



# Add Fully Connected Layer with 120 units

model.add(Dense(84, activation="tanh"))



#Add Fully Connected Layer with 10 units and activation function as 'softmax'

model.add(Dense(10, activation="softmax"))
model.summary()
model.compile(optimizer='adam', loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

model.fit(x=X_train, y=y_train, batch_size=32, epochs=30)
predictions = model.predict_classes(X_test,verbose=1)
sub = pd.read_csv('../input/digit-recognizer/sample_submission.csv')

submission = sub['ImageId']
pred = pd.DataFrame(data=predictions ,columns=["Label"])

DT = pd.merge(submission , pred, on=None, left_index= True,

    right_index=True)

DT.head()
DT.to_csv('submission.csv',index = False)