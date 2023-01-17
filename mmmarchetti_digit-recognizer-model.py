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
train_df = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test_df = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")



train_df.head()
print(train_df.shape)



y_train_df = train_df['label']

x_train_df = train_df.drop('label', axis=1)

print(y_train_df.shape)



# del train_df['label']

print(x_train_df.shape)
import matplotlib.pyplot as plt

import seaborn as sns



# y_train.value_counts()

sns.countplot(y_train_df)
unique_labels = y_train_df.unique()

num_labels = len(unique_labels)

print(num_labels)

print(unique_labels)
print(x_train_df.shape)

print(y_train_df.shape)

print(test_df.shape)
from sklearn.model_selection import train_test_split



input_shape = (28, 28, 1)



X_train, X_val, y_train, y_val = train_test_split(x_train_df, y_train_df, test_size=0.3, random_state=0)

X_train = X_train.values.reshape(-1, 28, 28, 1)

X_val = X_val.values.reshape(-1, 28, 28, 1)

X_test = test_df.values.reshape(-1, 28, 28, 1)



y_train = np.asarray(y_train)

y_val = np.asarray(y_val)



print(X_train.shape)

print(X_val.shape)

print(X_test.shape)
# feature scaling (dot in 255. means its a float value)

X_train = X_train / 255.

X_val = X_val / 255.

X_test = X_test / 255.
from __future__ import absolute_import, division, print_function, unicode_literals



import tensorflow as tf

from tensorflow.keras import layers



tf.keras.backend.clear_session()  # For easy reset of notebook state.
model = tf.keras.Sequential()



# add convolutional layer to the model with relu activation

# 32 convolution filters used each of size 3x3

# the input shape is (28,28,1)

model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))

model.add(layers.MaxPooling2D(2))

model.add(layers.Conv2D(64, (3,3), activation='relu'))

model.add(layers.MaxPooling2D(2))

model.add(layers.Conv2D(128, (3,3), activation='relu'))

model.add(layers.MaxPooling2D(2))

model.add(layers.Flatten())

model.add(layers.Dense(32, activation='relu'))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(32, activation='relu'))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(num_labels, activation='softmax'))





model.summary()
model.compile(loss='sparse_categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
num_epochs = 50

# batch size -Total number of training examples present in a single batch.

batch_size = 64

history = model.fit(X_train, y_train, epochs=num_epochs,

                    batch_size=batch_size, 

                    verbose=1,

                    validation_data=(X_val, y_val))
score = model.evaluate(X_val, y_val, verbose=0)

print('Test loss:', score[0]) #Test loss: 0.0296396646054

print('Test accuracy:', score[1]) #Test accuracy: 0.9904
predictions = model.predict(X_val)



print(np.argmax(predictions[0]))

print(y_val[0])
final_predictions = model.predict(X_test)

final_predictions = list(map(lambda x : np.argmax(np.round(x)), final_predictions))

final_predictions[:10]
predicted_labels = pd.Series(final_predictions, name="Label")

image_id = pd.Series(range(1, len(predicted_labels)+1),name="ImageId")



results = pd.concat([image_id,predicted_labels],axis=1)



results.to_csv("MNIST.csv",index=False)