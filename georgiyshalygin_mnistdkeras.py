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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
%matplotlib notebook
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv').astype('float32')
task = pd.read_csv('/kaggle/input/digit-recognizer/test.csv').astype('float32')
Y = to_categorical(train['label'], num_classes=10)
y_train, y_test = Y[:40000], Y[40000:]
X = train.drop(labels = ["label"],axis = 1) / 255
x_train, x_test = X[:40000], X[40000:]
model = Sequential([
    layers.Dense(800, activation='relu', input_shape=(28*28,)),
    layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss=keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

history = model.fit(X, Y, batch_size=64, epochs=6, shuffle=True)

test_scores = model.evaluate(x_test, y_test, verbose=2)
test_scores
pred = model.predict(task)
pred
pred = np.argmax(pred, axis=1)
pred
pred = pd.DataFrame(zip(list(range(1, len(pred) + 1)),
                        pred))
pred.to_csv('ans', index=False, header=['ImageId', 'Label'])
