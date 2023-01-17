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
fahrenheit = np.random.random((100, 1)) * 100

celcius = (fahrenheit - 32) / 1.8
print ("Fahrenheit: {}".format(fahrenheit[0]))

print ("Celcius: {}".format(celcius[0]))
#Lets split this data up into a train and test set

train_split = 0.7



train_f = fahrenheit[0:int(len(fahrenheit)*train_split)]

train_c = celcius[0:int(len(celcius)*train_split)]



test_f = fahrenheit[len(train_f):]

test_c = celcius[len(train_c):]
from tensorflow.keras import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.callbacks import EarlyStopping

from matplotlib import pyplot as plt
#Early stop the training

callball = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
perceptron = Sequential()

perceptron.add(Dense(1, input_dim=1, activation = 'linear'))

perceptron.compile(loss = 'mse', learning_rate=0.001)
training = perceptron.fit(train_c, train_f, epochs=10000, verbose=False, callbacks=[callball])
perceptron.get_weights()
plt.plot(training.history['loss'])
perceptron.predict([20])
perceptron = Sequential()

perceptron.add(Dense(3, input_dim=1, activation = 'linear'))

perceptron.add(Dense(1, activation = 'linear'))

perceptron.compile(loss = 'mse', learning_rate=0.01)
training = perceptron.fit(train_f, train_c, epochs=8000, verbose=False)
plt.plot(training.history['loss'])
perceptron.predict([25])
print (perceptron.layers[0].get_weights())

print (perceptron.layers[1].get_weights())
#Begin format of the classification NN model with classification data generate from sklearn

from sklearn.datasets import make_classification, make_circles, make_blobs



X, y = make_circles(n_samples=1000, noise=0.05, factor=0.5)



colors=['red', 'blue']



plt.scatter(X[:, 0], X[:, 1], c=y)