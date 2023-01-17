from keras.models import Sequential

from keras.layers import Dense

import numpy
seed = 9

numpy.random.seed(seed)
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from pandas import read_csv



filename = '/kaggle/input/BBCN.csv'

dataframe = read_csv(filename)
array = dataframe.values
X = array[:,0:11] 

Y = array[:,11]
dataframe.head()
model = Sequential()

model.add(Dense(12, input_dim=11, bias_initializer='uniform', activation='relu'))

model.add(Dense(8, bias_initializer='uniform', activation='relu'))

model.add(Dense(1, bias_initializer='uniform', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=200, batch_size=30)
scores = model.evaluate(X, Y)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
