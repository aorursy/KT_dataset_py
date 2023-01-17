# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import tensorflow as tf

from tensorflow import keras

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
inputs = pd.read_csv('../input/digit-recognizer/train.csv')

Labels = np.array(inputs['label'])

pixels = np.array(inputs)[:,1:]



model = tf.keras.Sequential([

    keras.layers.Dense(2048,activation = tf.nn.relu),

    keras.layers.Dense(1024, activation = tf.nn.relu),

    keras.layers.Dense(512,activation = tf.nn.relu),

    keras.layers.Dense(256, activation = tf.nn.relu),

    keras.layers.Dense(128,activation = tf.nn.relu),

    keras.layers.Dense(10,activation = tf.nn.softmax)

])
from tensorflow.keras.optimizers import RMSprop

model.compile(optimizer=tf.keras.optimizers.Adam(),

              loss=tf.keras.losses.SparseCategoricalCrossentropy(),

              metrics = ['accuracy'])
history = model.fit(pixels[:40000,:], Labels[:40000], epochs = 30)
model.evaluate(pixels[40000:42000,:], Labels[40000:42000])
test_input_pd = pd.read_csv('../input/digit-recognizer/test.csv')

test_input = np.array(test_input_pd)

predictions = []

test_mid_output = list(model.predict(test_input))

for i in (test_mid_output):

    j = list(i)

    output = j.index(max(j))

    predictions.append(output)

imageId = list(range(1,28001))

out = pd.DataFrame({'ImageId': imageId, 'Label': predictions})

out.to_csv('submission.csv', index=False)