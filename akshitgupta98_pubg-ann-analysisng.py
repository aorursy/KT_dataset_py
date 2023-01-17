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
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
train = pd.read_csv("../input/train_V2.csv")
test = pd.read_csv("../input/test_V2.csv")
test_id=test.Id
train = train.iloc[: , [3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,22,23,24,25,26,27,28]].values
test = test.iloc[:,[3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,22,23,24,25,26,27]].values
test

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN" , strategy ="median" , axis = 0)
imputer = imputer.fit(train[:,:])
train[:,:]=imputer.transform(train[:,:])
imputer = imputer.fit(test[:,:])
test[:,:]=imputer.transform(test[:,:])
xtrain = train[:,0:24]
ytrain = train[:,-1]
xtest = test[:,:]

xtrain = np.array(xtrain.round())
ytrain = np.array(ytrain.round())
xtest = np.array(xtest.round())
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.transform(xtest)
import keras
from keras.models import Sequential
from keras.layers import Dense
# Initialising the ANN
classifier = Sequential()

xtrain.shape


# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu', input_dim = 24))

# Adding the second hidden layer
classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
xtrain.shape
ytrain.shape
# Fitting the ANN to the Training set
classifier.fit(xtrain, ytrain, batch_size = 10, epochs = 1)
# Predicting the Test set results
ypred = classifier.predict(xtest)
ypred = ypred.flatten()
submission=pd.DataFrame({'Id':test_id,'winPlacePerc':ypred})
submission.to_csv('submission.csv', index=False)