# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Breast_cancer_data.csv')

data.head(10)
data.info()
smoothness = data['mean_smoothness'] > 1.0

smoothness.values
X = data.iloc[:,0:5].values

Y = data.iloc[:,5].values
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=0)
from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

X_train = sc.fit_transform(x_train)

X_test = sc.fit_transform(x_test)
import keras

from keras.models import Sequential

from keras.layers import Dense



classifier = Sequential()



# Hidden layers and input layer are  added.

classifier.add(Dense(3, kernel_initializer="uniform", activation = 'relu', input_dim = 5))



classifier.add(Dense(3, kernel_initializer="uniform", activation = 'relu'))



# Output layer

classifier.add(Dense(1, kernel_initializer="uniform", activation = 'sigmoid'))



classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



classifier.fit(X_train, y_train, epochs = 50)



# Prediction

y_pred = classifier.predict(X_test)
y_pred
y_pred = (y_pred > 0.5)



from sklearn.metrics import confusion_matrix



cm = confusion_matrix(y_test, y_pred)

cm