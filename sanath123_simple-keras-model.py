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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data_df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

print(data_df.shape)
test_df = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

test_df.shape
X_data = data_df[data_df.columns[1:]]

Y_data = data_df['label']
print(X_data.columns[X_data.isnull().sum() > 0])

print(Y_data[Y_data.isnull()])


X_data = X_data / 255.0

test_df = test_df / 255.0
# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)

X_data = X_data.values.reshape(-1,28,28,1)

test_df = test_df.values.reshape(-1,28,28,1)
# Split the train and the validation set for the fitting

random_seed = 10

X_train, X_val, Y_train, Y_val = train_test_split(X_data, Y_data, test_size = 0.2, random_state=random_seed)

print(X_train.shape)

print(X_val.shape)
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(16, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(10, activation='softmax')

])
model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])

model.summary()
history = model.fit(X_train, Y_train, epochs=5, validation_data=(X_val, Y_val))
model.save('model.imageclassifier')
results = model.predict(test_df)
results = pd.Series(np.argmax(results,axis = 1))

results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("sumbission.csv",index=False)