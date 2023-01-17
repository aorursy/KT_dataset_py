import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#Import models from scikit learn module:

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from tensorflow.keras import optimizers

from keras.optimizers import Adam

from keras.datasets import imdb

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers.embeddings import Embedding

from keras.preprocessing import sequence

from sklearn import datasets
cancer = datasets.load_breast_cancer()
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(cancer.data,cancer.target,test_size = 0.3)
model = Sequential()

model.add(Dense(10,input_shape=(30,), activation='sigmoid'))

model.add(Dense(10,activation='sigmoid'))

model.add(Dense(10,activation='sigmoid'))

model.add(Dense(1,activation='sigmoid'))
sgd = optimizers.SGD(lr = 0.01)

model.compile(loss='binary_crossentropy',optimizer=sgd,

metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train, batch_size=50,epochs=100, verbose = 1)
results = model.evaluate(x_test, y_test)

results