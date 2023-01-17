
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
import matplotlib.pyplot as plt
%matplotlib inline 
df_traine = pd.read_csv('../input/train.csv')
df_traine.head()
df_traine.shape
images = df_traine.drop('label',axis=1).values
images = images.reshape(-1,28,28)
plt.imshow(images[2])
from keras.layers import Conv2D, Activation,Input,Dense
from keras.models import Model
def build_model():
    input_layer = Input((28,28,1))
    conv1 = Conv2D(filters=16, kernel_size=(3,3),strides=(1,1))(input_layer)
    conv2 = Conv2D(filters=16, kernel_size=(3,3),strides=(1,1))(conv1)
   
    dense1 = Dense(units=10, name='dense1')(conv2)
    dense1 = Activation('relu', name='activation_dense1')(dense1)
    
    model = Model(input_layer,dense1)
    return model
build_model().summary()