# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import tensorflow as tf
from tensorflow import keras

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#getting the data
trueData = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/True.csv')
falseData = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/Fake.csv')

trueTitles = trueData['title']
falseTitles = falseData['title']

trueTitlesArr = trueTitles.to_numpy()
falseTitlesArr = falseTitles.to_numpy()

trueDataY = np.ones(len(trueTitles))
falseDataY = np.zeros(len(falseTitles))

dataX = np.concatenate([trueTitles, falseTitles])
dataY = np.concatenate([trueDataY, falseDataY])

aux = list(zip(list(dataX), list(dataY)))

random.shuffle(aux)

dataX, dataY = zip(*aux)
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
vectorizer = TextVectorization(output_mode="int")

vectorizer.adapt(dataX)

integer_data = vectorizer(dataX)
print(integer_data)
from tensorflow.keras import layers
inputs = keras.Input(shape=(43))
x = layers.Dense(100, activation='relu')(inputs)
x = layers.Dense(100, activation='relu')(x)
x = layers.Dense(50, activation='relu')(x)
x = layers.Dense(10, activation='relu')(x)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
model.summary()
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
import tensorflow_datasets as tfds
batch_size = 1000

x_train = tfds.as_numpy(integer_data)
x_train = list(x_train)
x_train = x_train[:int(len(x_train)/100*80)]
y_train = list(dataY)

for i in range(len(y_train)):
    if y_train[i] == 1:
        y_train[i] = 0.999999

y_train = y_train[:int(len(dataY)/100*80)]

x_train = np.array(x_train)
y_train = np.array(y_train)

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=10)
model.save('path_to_my_model.h5')
new_model = keras.models.load_model('path_to_my_model.h5')
new_predictions = new_model.predict(x_train[0])
print(new_predictions, y_train[0])