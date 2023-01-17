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
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
%matplotlib inline
train = pd.read_csv('../input/train.csv')
print(train.head())
test  = pd.read_csv('../input/test.csv')
X = train.iloc[:, 1:]
print(X.shape)

y = train.iloc[:, :1]
print(y.shape)

class_ = y['label'].unique()
print(class_)
#normalizing
#We scale these values to a range of 0 to 1 before feeding to the neural network model. 
#For this, cast the datatype of the image components from an integer to a float, and divide by 255. 
X = X.astype('float32')/255
test = test.astype('float32')/255

X = X.values
test = test.values

#reshape
w, h = 28, 28
X = X.reshape(X.shape[0], w, h)
test = test.reshape(test.shape[0],w, h)



#Displaying the image
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X[i], cmap=plt.cm.binary)
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(512, activation = tf.nn.relu),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(512, activation = tf.nn.relu),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation = tf.nn.softmax)
])
model.compile(optimizer = keras.optimizers.RMSprop(lr=0.001),
             loss ='sparse_categorical_crossentropy',
             metrics = ['accuracy'])
model.fit(X, y, epochs = 20, batch_size=64)
model.summary()
predictions = model.predict_classes(test, verbose=0)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("mnist_tfkeras.csv", index=False, header=True)
