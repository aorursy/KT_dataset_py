# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

print(pd.__version__)





import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("/kaggle/working"))



# Any results you write to the current directory are saved as output.
test = pd.read_csv("../input/test.csv")

train = pd.read_csv("../input/train.csv")



train_labels = train['label'].values

train_input = train.drop('label', axis=1).values



test_input = test.values


for i in range(1, 5):

    plt.subplot(2,2, i)

    plt.imshow(train_input[i].reshape((28,28)))

    plt.title(train_labels[i])

plt.show()
model = tf.keras.Sequential([

    tf.keras.layers.Dense(128, activation=tf.nn.relu),

    tf.keras.layers.Dense(10, activation=tf.nn.softmax)

])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
train_input, test_input = train_input / 255.0, test_input / 255.0
model.fit(train_input, train_labels, epochs=5)
predictions = model.predict(test_input)

pred_labels = np.argmax(predictions, axis=1)

submission = pd.DataFrame({'ImageId': range(1,28001), 'Label': pred_labels})

submission.to_csv("submission.csv", index=False)