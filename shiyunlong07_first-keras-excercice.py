# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
y_train = train['label']
x_train = train.drop(labels = ['label'], axis = 1)
x_train = x_train/225.0

test = test/225.0
x_train = x_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
from sklearn.model_selection import train_test_split



x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, random_state = 1) 
import matplotlib.pyplot as plt



fig = plt.figure(figsize =(20,20))

for i in range(6):

    ax = fig.add_subplot(1, 6, i+1, xticks = [], yticks = [])

    ax.imshow(x_train[i][:,:,0], cmap = 'gray')
from keras.models import Sequential, Model

from keras.utils import to_categorical



from keras.layers import Dense, Input, Embedding, LSTM

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Flatten
x_train=x_train.reshape(-1,784)

x_val=x_val.reshape(-1,784)
model = Sequential()
model.add(Dense(units = 784, activation = 'relu', input_dim = 784))

model.add(Dense(units = 10, activation = 'softmax'))
model.summary()
model.compile(loss = 'categorical_crossentropy',

             optimizer = 'sgd',

             metrics = ['accuracy'])
print(x_train.shape)

print(x_val.shape)

print(y_train.shape)

print(y_val.shape)
y_train = to_categorical(y_train)

y_val = to_categorical(y_val)
model.fit(x_train, y_train, epochs = 30, batch_size = 32)
score = model.evaluate(x_val, y_val, batch_size = 128)

print('loss: ', score[0])

print('accuracy: ', score[1])
result = model.predict_classes(x_val, batch_size = 128)

result
result = pd.Series(result, name = 'label')

result
submission = pd.concat([pd.Series(range (1,28001), name = "ImageId"), result], axis = 1)
submission.to_csv("First Keras excercice.csv", index = False)