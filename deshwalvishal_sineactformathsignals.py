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
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 
import numpy as np
import keras.optimizers
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from keras.layers import Dense
t=np.linspace(-3*np.pi,3*np.pi,1024*4)
y=np.sin(np.pi * t/3)
plt.plot(t,y)
max2=np.max(y)
min2=np.min(y)
min3=-1*min2
range2=max2+min3
n=256
print(max2, min2, range2)
step=range2/n
new_data2=np.zeros((y.shape[0]))
print(step)
for k in range(n):
    a=min2+k*step
    b=min2+(k+1)*step
    for j in range(y.shape[0]):
        if(y[j]>=a and y[j]<=b):
            new_data2[j]=a
plt.plot(t,new_data2)
s=int(new_data2.shape[0])
indices=np.random.permutation(s) 
j=int(indices.shape[0]*0.3)
test_indices=indices[:j]
train_indices=indices[j:]
print(test_indices.shape)

y_train= new_data2[train_indices]
y_test= new_data2[test_indices]
X_train= t[train_indices]
X_test= t[test_indices]


from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
def custom_activation(x):
  return ((tf.math.sin(x)))
get_custom_objects().update({'custom_activation':Activation(custom_activation)})
model = keras.Sequential()
model.add(Dense(8, input_dim = 1))
model.add(Activation(custom_activation, name='SinActivation'))
model.add(Dense(1,activation="sigmoid", name="last",kernel_initializer = "normal"))
model.compile(loss='mean_squared_error',optimizer='adam') 
epochs = 50
history = model.fit(t, new_data2,epochs=epochs,validation_data=(X_test,y_test),batch_size=128,verbose=2) 
pred = model.predict(X_test)
new_data1 = np.zeros(new_data2.shape)
print(new_data1.shape)
new_data1 = np.reshape(new_data1,(new_data1.shape[0],1))
y_test= np.reshape(y_test,(y_test.shape[0],1))
new_data1[test_indices] = y_test
y_train = np.reshape(y_train,(y_train.shape[0],1))
new_data1[train_indices] = y_train
new_data3 = np.zeros(new_data2.shape)
print(new_data3.shape)
new_data3 = np.reshape(new_data3,(new_data3.shape[0],1))
pred= np.reshape(pred,(pred.shape[0],1))
new_data3[test_indices] = pred
y_train = np.reshape(y_train,(y_train.shape[0],1))
new_data3[train_indices] = y_train
plt.plot(t,new_data3,'o')