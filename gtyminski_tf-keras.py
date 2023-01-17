# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.import tensorflow as tf
import tensorflow as tf

data_dir = '../input/'

import os
print(os.listdir("../input"))

# load csv file
def load_data(row_nums):
    train = pd.read_csv(data_dir + 'train.csv').values
    x_test = pd.read_csv(data_dir + 'test.csv').values
    print(train.shape)

    x_train = train[:row_nums, 1:]
    y_train = train[:row_nums, 0]
    return x_train, y_train, x_test

x_train, y_train, x_test = load_data(42000)
x_train, x_test = x_train / 255.0, x_test / 255.0
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(784, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
optimizer = tf.keras.optimizers.Adam(
    lr=0.0001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=None,
    decay=0.0,
    amsgrad=True)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100)
# model.evaluate(x_test, y_test)

### Export predictions to file for KAGGLE verification
y_pred = model.predict(x_test)

# output the prediction result
y_pred_max = y_pred.argmax(1)
pd.DataFrame({"ImageId": list(range(1,len(y_pred)+1)), "Label": y_pred_max}).to_csv('./Digit_Recogniser_Result.csv', index=False, header=True)
print(y_pred_max, len(y_pred_max))

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# import os
# print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
