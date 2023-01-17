# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow import keras

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
print(tf.__version__)


sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")

train = pd.read_csv("../input/digit-recognizer/train.csv")
train.head(2)
X_train = train.drop(['label'], axis=1)
X_train.head(2)
Y_train = train['label']
X_train=X_train.astype('float32')/255.0

X_train = X_train.values.reshape(-1,28,28,1)

plt.imshow(X_train[1][:,:,0])

plt.show()
test=test.astype('float32')/255.0

test = test.values.reshape(-1,28,28,1)
model = keras.Sequential([

    keras.layers.Flatten(input_shape=(28,28,1)),

    keras.layers.Dense(128, activation="relu"),

    keras.layers.Dense(10, activation="softmax")

])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, Y_train, epochs=5)
prediction = model.predict(test)
prediction.shape
predict_list = []

for i in range(28000):

    predict_list.append(np.argmax(prediction[i]))
plt.imshow(test[2][:,:,0])

plt.show()
plt.imshow(test[2][:,:,0])

plt.show()
sample_submission
np.argmax(prediction[0])
my_submission = pd.DataFrame({'ImageId':sample_submission.loc[:,'ImageId'], 'Label':predict_list})
my_submission.to_csv('my_submission.csv', index=False)