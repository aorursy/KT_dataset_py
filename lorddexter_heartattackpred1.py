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
dataset = pd.read_csv('/kaggle/input/heart-attack-prediction/data.csv')

dataset.head()
dataset.replace('?', np.nan, inplace = True)



X = dataset.iloc[:, 0:13].values

Y = dataset.iloc[:, 13].values



# Fixing missing data

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = np.nan, strategy = 'mean', axis = 0)

imputer = imputer.fit(X[:, 0:13])

X[:, 0:13] = imputer.transform(X[:, 0:13])



# Normalizing the data

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X = sc_X.fit_transform(X)



# Splitting dataset into training and test

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.4)



# Building a ANN

import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.utils import to_categorical

from keras import regularizers

from keras.layers import Dropout



Y_train = to_categorical(Y_train)

Y_test = to_categorical(Y_test)



classifier = Sequential([Dense(units = 8, activation = 'relu', input_shape = (13, ), kernel_regularizer = regularizers.l2(0.01)), 

                         Dropout(0.2),

                         Dense(units = 2, activation = 'softmax'),

                         ])



classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])



classifier.fit(X_train, Y_train, validation_data = (X_test, Y_test), epochs = 500)



# Evaluating the model

classifier.evaluate(X_test, Y_test)[1]