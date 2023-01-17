# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from tensorflow import keras

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)

ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)



#Define the model

model=keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])



#Compile the model

model.compile(optimizer='sgd',loss='mean_squared_error')



#Training the NN

model.fit(xs,ys,epochs=500)



print(model.predict([15.0]))