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
from sklearn.model_selection import train_test_split

from tensorflow import keras

i_row,i_cols = 28,28

num_class = 10

def prepare_data(raw_data):

    y = raw_data[:,0]

    out_y = keras.utils.to_categorical(y, num_class)

    x = raw_data[:,1:]

    num_images = raw_data.shape[0]

    out_x = x.reshape(num_images,i_row,i_cols,1)

    out_x = out_x/255

    return out_x,out_y



input_file = "../input/train.csv"

input_data = np.loadtxt(input_file,delimiter =',',skiprows =1)

x,y = prepare_data(input_data)

from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Flatten,Conv2D



model = Sequential()
model.add(Conv2D(40,kernel_size=(3,3),activation = 'relu',input_shape = (i_row,i_cols,1)))

model.add(Conv2D(25,kernel_size =(3,3),activation = 'relu'))

model.add(Conv2D(100,kernel_size =(3,3),activation = 'relu'))

model.add(Flatten())

model.add(Dense(100,activation = 'relu'))

model.add(Dense(num_class,activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])
model.fit(x,y,batch_size = 100,epochs = 4,validation_split = 0.2)