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
df = pd.read_csv("../input/Admission_Predict.csv",sep = ",")
df.head()
df.columns
features = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP',

       'LOR ', 'CGPA', 'Research']

#df[features]
X = df[features]
y = df['Chance of Admit ']
type(X)
X.shape
type(y)
from __future__ import absolute_import, division, print_function

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

l0 = tf.keras.layers.Dense(units=1, input_shape=[7])  
model = tf.keras.Sequential([l0])
model.compile(loss='mean_squared_error',

              optimizer=tf.keras.optimizers.Adam(0.1))
history = model.fit(X, y, epochs=5, verbose=False)

print("Finished training the model")
import matplotlib.pyplot as plt

plt.xlabel('Epoch Number')

plt.ylabel("Loss Magnitude")

plt.plot(history.history['loss'])
plt.plot(model.predict(X[0:100]), y[0:100], "^")