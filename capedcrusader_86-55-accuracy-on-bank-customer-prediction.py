# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import matplotlib.pyplot as plt

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
pd.read_csv('/kaggle/input/bank-customers/Churn Modeling.csv')
dataset = pd.read_csv('/kaggle/input/bank-customers/Churn Modeling.csv')

X = dataset.iloc[:, 3:13].values

y = dataset.iloc[:, 13].values
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

le_1 = LabelEncoder()

X[:,1] = le_1.fit_transform(X[:,1])

le_2 = LabelEncoder()

X[:,2] = le_2.fit_transform(X[:,2])

ohe = OneHotEncoder(categorical_features = [1])

X = ohe.fit_transform(X).toarray()

X = X[:, 1:]
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)
import keras

keras.__version__
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout
classifier = Sequential()
#Adding the input layer and first layer

classifier.add(Dense(activation = 'selu', input_dim = 11, units = 6, kernel_initializer = 'lecun_normal'))

classifier.add(Dropout(rate = 0.1))

#Adding the second hidden layer

classifier.add(Dense(activation='selu', units = 6, kernel_initializer="lecun_normal"))

classifier.add(Dropout(rate = 0.1))

#Adding the third hidden layer

classifier.add(Dense(activation = 'selu', units = 6, kernel_initializer = 'lecun_normal'))

classifier.add(Dropout(rate  =0.1))

#Adding the fourth hidden layer

classifier.add(Dense(activation = 'selu', units = 6, kernel_initializer = 'lecun_normal'))

classifier.add(Dropout(rate  =0.1))

#Adding the output layer

classifier.add(Dense(activation = 'sigmoid', units = 1, kernel_initializer = 'uniform'))
classifier.compile(optimizer = 'nadam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train,y_train,batch_size = 32, epochs = 100)
y_pred = classifier.predict(X_test)

y_pred = (y_pred>0.5)
y_pred
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)

cm
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test,y_pred)

accuracy