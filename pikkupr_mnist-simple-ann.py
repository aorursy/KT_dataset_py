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
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cross_validation import train_test_split
dataset = pd.read_csv('../input/train.csv')
dataset.head(5)
y = dataset['label']
X = dataset.copy()
del X['label']

X = X/255.00
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=41)
x_train.shape
x_train = x_train.values
x_test = x_test.values
y_train = y_train.values
y_test = y_test.values
model = tf.keras.Sequential([
        tf.keras.layers.Dense(128),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.train.AdamOptimizer(),
    metrics=['accuracy']
)
model.fit(x_train, y_train, epochs=10)
model.evaluate(x_test, y_test)
pd.read_csv('../input/sample_submission.csv').head(2)
prediction_dataset = pd.read_csv("../input/test.csv")
prediction_dataset.head(2)
prediction_dataset = prediction_dataset/255.0
model.fit(X, y, epochs=10)
predictions = model.predict(prediction_dataset)
predictions = [np.argmax(row) for row in predictions]
predictions[:2]
submission_data =  pd.DataFrame()
submission_data['Label'] = predictions
submission_data.index.name = "ImageId"
submission_data.reset_index(inplace=True)
submission_data['ImageId'] = submission_data['ImageId'] + 1
submission_data.head(5)
submission_data.to_csv('submission1.csv', index=False)