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
from keras.layers import Input, Dense, Dropout
from keras.models import Model
import numpy as np

xs_and  = np.array([[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]], dtype = float)
ys_and = np.array([0.0,0.0,0.0,1.0], dtype = float)
#Firstly  ,we will define input layers and we have 2 shape
inputs = Input(shape=(2,))

#and later we will define a hidden layer with 20 neuron and define activation function as 'relu'
#then connect with imputs layer
layer1 = Dense(20, activation='relu')(inputs)

#we add droupout layer . The aim of this is that we try to prvent overfitting
#then we connect this layer with layer1
layer_dropout = Dropout(rate = 0.1)(layer1)

# we define second layer with 20 neuron and connect with layer_dropout
layer2 = Dense(20, activation='relu')(layer_dropout)

#Last layer will be output and we define activation function as sigmoid 
#then connect this layer with layer2
predictions = Dense(1, activation='sigmoid')(layer2)

# In this step , we will create model and we will just inputs and outputs .
model = Model(inputs=inputs, outputs=predictions)

# we use compile part to initiliaze our parameters and design our model
# loss function 
#optimizer -- can be adam, sgd or etc
# we define metrics as accuracy to see how our model works properly
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# starts training our model 
model.fit(xs_and, ys_and, batch_size=10, epochs=50, verbose = 0) 

# make predicts
# we don't have big data to split our data training and test so we used training data
model.predict(xs_and)