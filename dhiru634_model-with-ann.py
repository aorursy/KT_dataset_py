# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
xtrain=np.load("../input/Training.npy")

ytrain=np.load("../input/TrainingY.npy")

xtest=np.load("../input/Testing.npy")

ytest=np.load("../input/TestY.npy")
# Showing the samples of the training set take one image from each class  

fig=plt.figure(figsize=(30,12))

m=1

for i in range(1,117000,3000):

    a=fig.add_subplot(3,13,m)

    m=m+1

    plt.imshow(xtrain[i]) 
import tensorflow as tf

from tensorflow import keras
model=tf.keras.Sequential([keras.layers.Flatten(input_shape=(32,32)),

                          keras.layers.Dense(784,activation='tanh'),

                          keras.layers.Dense(784,activation='tanh'),keras.layers.Dense(784,activation='tanh')

                           ,keras.layers.Dense(39,activation='softmax')])

model.compile(optimizer='Adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(xtrain,ytrain,epochs=60,verbose=1,batch_size=500)
score=model.evaluate(xtest,ytest,verbose=1)
print(score)