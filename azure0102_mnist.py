# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_orig = pd.read_csv('../input/train.csv')

test_orig = pd.read_csv('../input/test.csv')
train_data = train_orig.drop(columns = ["label"], axis = 1)

train_data = 1.0*train_data / 255 # normalization

train_label = train_orig["label"]

# X_train, X_val, Y_train, Y_val = train_test_split(train_data, train_label, test_size = 0.1, random_state = 2)

test_data = 1.0*test_orig / 255
train_label[0]
train_data = train_data.values.reshape(-1,28,28,1)

test_data = test_data.values.reshape(-1,28,28,1)
model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(64, (3,3), activation = "relu", input_shape = (28,28,1), padding = "same"),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(rate = 0.25),

    tf.keras.layers.Conv2D(64, (3,3), activation = "relu", padding = "same"),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(rate = 0.25),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation = "relu"),

    tf.keras.layers.Dropout(rate = 0.25),

    tf.keras.layers.Dense(10, activation = "softmax")

])
model.summary()
model.compile(optimizer='adam', loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
history = model.fit(train_data, train_label, batch_size = 64, epochs = 100, validation_split = 0.1, verbose = 1)
import matplotlib.pyplot as plt



plt.plot(history.history["acc"])

plt.plot(history.history["val_acc"])

plt.title("model accuracy")

plt.show()
predict = model.predict(test_data)
predict = np.argmax(predict, axis = 1)
submission = pd.concat([pd.Series(range(1 ,28001) ,name = "ImageId"), pd.Series(predict, name="Label")],axis = 1)
submission.to_csv('submission.csv', index=False)
submission.head()