import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import scipy as sc

import random

import time

from sklearn.utils import shuffle



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical



from keras.models import Sequential

from keras.layers import Dense

in_data = pd.read_csv('../input/Iris.csv')

in_data.head()
in_data = in_data.drop(['Id'], axis = 1)

in_data
in_data = shuffle(in_data)

X = in_data.drop(['Species'], axis = 1)

X = np.array(X)

Y = np.array(in_data['Species'])

X[:10], Y[:10]
l_encode = LabelEncoder()

l_encode.fit(Y)

Y = l_encode.transform(Y)

Y = to_categorical(Y)

Y
train_x, test_x, train_y, test_y = train_test_split(X,Y, test_size = 0.3, random_state = 0)

train_x.shape, train_y.shape, test_x.shape, test_y.shape
in_dim = len(in_data.columns)-1



model = Sequential()

model.add(Dense(8, input_dim = in_dim, activation = 'relu'))

model.add(Dense(10, activation = 'relu'))

model.add(Dense(10, activation = 'relu'))

model.add(Dense(10, activation = 'relu'))

model.add(Dense(3, activation = 'softmax'))



model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.fit(train_x, train_y, epochs = 15, batch_size = 5)

scores = model.evaluate(test_x, test_y)



for i, m in enumerate(model.metrics_names):

    print("\n%s: %.3f"% (m, scores[i]))
test_size = 10

pred = model.predict_classes(test_x[:test_size])

pred_ = np.argmax(to_categorical(pred), axis = 1)

pred_ = l_encode.inverse_transform(pred_)



true_y = l_encode.inverse_transform(np.argmax(to_categorical(test_y[:test_size]), axis = 1)[:,1])



for i,j in zip(pred_, true_y):

    print("Predicted: {}, True: {}".format(i, j))