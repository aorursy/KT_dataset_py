# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tensorflow import keras

import matplotlib.pyplot as plt

import tensorflow as tf

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#import os

#print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
x = np.array([-1.0,0.0,1.0,2.0,3.0,4.0], dtype = float)

y = np.array([-3.0,-1.0,1.0,3.0,5.0,7.0], dtype = float)
plt.plot(x,y)

plt.title('Visualizing relation between x and y')

plt.ylabel('Dependent variable (y)')

plt.xlabel('Independent variable (x)')

plt.show()
model = keras.Sequential([keras.layers.Dense(units =1, input_shape=[1])])

model.compile(optimizer = 'sgd', loss = 'mean_squared_error')
model.fit(x,y, epochs = 100)



print("predicted Answer : " ,  model.predict([10.0]))
celsius_q = np.array([-40,-10,0,8,15,22,38], dtype = float) #x variables



fahrenheit_a = np.array([-40,14,32,46,59,72,100], dtype = float) #y variables



#lets print these



for f,c in enumerate (celsius_q):

  print("{} degree cesius = {} degree fahrenheit".format(c,fahrenheit_a[f]))
ten = tf.keras.layers.Dense(units = 1, input_shape = [1])



model = tf.keras.Sequential([ten])
model.compile(loss = 'mean_squared_error',

             

              optimizer = tf.keras.optimizers.Adam(0.1) #0.1 here is the learning rate

             )
train = model.fit(celsius_q, fahrenheit_a, epochs = 100, verbose = False )



print("training has been finished")
plt.xlabel('Epoch Number')



plt.ylabel('Loss Magnitude')



plt.plot(train.history['loss'])
print(model.predict([100.0]))
print("These are the layer variables: {}".format(ten.get_weights()))