# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import tensorflow as tf
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

print(train_df.shape, test_df.shape)
X_train = train_df.iloc[:, 1:].values.reshape(-1, 28, 28, 1)

y_train = train_df.iloc[:, 0].values

X_test = test_df.values.reshape(-1, 28, 28, 1)

index = 5

plt.imshow(X_train[index].reshape(28,28))

print(y_train[index])
def to_onehot(labels):

    ones = tf.one_hot(labels, 10, axis=0)

    with tf.Session() as sess:

        ones = sess.run(ones)

    return ones



y_train = to_onehot(y_train).T

y_train[:, :10]
X_train = X_train / 255

X_test = X_test / 255
X_train.shape, X_test.shape, y_train.shape 
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(8, kernel_size=(4,4), strides=(1, 1), padding='same'))

model.add(tf.keras.layers.Activation('relu'))

model.add(tf.keras.layers.MaxPool2D(pool_size=(8, 8), strides=(8, 8), padding='same'))

model.add(tf.keras.layers.Conv2D(16, kernel_size=(2, 2), strides=(1,1), padding='same'))

model.add(tf.keras.layers.Activation('relu'))

model.add(tf.keras.layers.MaxPool2D(pool_size=(4, 4), strides=(4, 4), padding='same'))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=32, validation_split=0.1, epochs=50)
prediction = model.predict(X_test)
prediction = np.argmax(prediction, axis=1)
submission = pd.DataFrame({'ImageId':np.arange(1, X_test.shape[0] + 1), 

                          'Label': prediction})

from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "forest.csv"):  

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



# create a random sample dataframe

# create a link to download the dataframe

create_download_link(submission)



# ↓ ↓ ↓  Yay, download link! ↓ ↓ ↓ 