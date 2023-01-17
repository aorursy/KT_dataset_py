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
from keras.models import Sequential

from keras.layers import Dense

from keras.activations import hard_sigmoid

import numpy as np
dataset = np.loadtxt('/kaggle/input/pimaindiansdiabetescsv/pima-indians-diabetes.csv', delimiter=',')
X = dataset[:, 0:8]

Y = dataset[:, 8]
model = Sequential()

model.add(Dense(1, input_shape=(8,), activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, epochs=100, batch_size=25, verbose=1, validation_split=0.2)
# define the keras model

model = Sequential()

model.add(Dense(12, input_dim=8, activation='relu'))

model.add(Dense(8, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.summary()

# compile the keras model