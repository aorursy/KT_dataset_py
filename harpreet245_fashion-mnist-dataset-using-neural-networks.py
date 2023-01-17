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
fashion_train = pd.read_csv('/kaggle/input/fashion-mnist-dataset-test-train/fashion-mnist_train.csv')

fashion_test = pd.read_csv('/kaggle/input/fashion-mnist-dataset-test-train/fashion-mnist_test.csv')
fashion_train.head(10)
import matplotlib.pyplot as plt
temp = []

final_image = []

for i in range(1, 785):

    if i%29 == 0:

        final_image.append(temp)

        temp = []

    else:

        temp.append(fashion_train.iloc[0, i])
final_image = np.array(final_image)

plt.imshow(final_image)
xtrain = fashion_train.iloc[:, 1:] 

xtest = fashion_test.iloc[:, 1:]



ytrain = fashion_train.iloc[:, 0]

ytest = fashion_test.iloc[:, 0]
import tensorflow as tf

import keras



from keras import Sequential

from keras.layers import Dense
from keras.utils import to_categorical



ytrain = to_categorical(ytrain, 10)

ytest = to_categorical(ytest, 10)
model = Sequential()
model.add(Dense(128, activation = 'relu', init = 'uniform', input_dim = 784))

model.add(Dense(64, activation = 'relu', init = 'uniform'))

model.add(Dense(5, activation = 'relu', init = 'uniform'))

model.add(Dense(10, activation = 'softmax', init = 'uniform'))
model.compile(optimizer= 'adam', metrics = ['accuracy'],

             loss = 'binary_crossentropy')
model.fit(xtrain, ytrain, epochs = 25)
model.evaluate(xtest, ytest)