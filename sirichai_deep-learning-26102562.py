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
# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



# Importing the dataset

dataset = pd.read_csv('../input/churn-modelling/Churn_Modelling.csv')

X = dataset.iloc[:, 3:13].values

y = dataset.iloc[:, 13].values
dataset.head()
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder



transformer = ColumnTransformer(

    transformers=[

        ("OneHot",           #Just

        OneHotEncoder(),

        [1,2]

        )

    ]

)



X2 = transformer.fit_transform(X)

print(X2.shape)

print(X2[0:20,:])
X = np.concatenate((X[:,0:1],X[:,3:10], X2[:,1:4]), axis=1)

print(X.shape)

print(X[0:5, :])
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
# Importing Keras libraries and packages

import keras

from keras.models import Sequential

from keras.layers import Dense



# Initialising the ANN

classifier = Sequential()
# Adding the input layer and the first hidden layer

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer

classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



#Adam คือการหาทิศทางและlearning rate ที่ดีที่สุด

#binary_crossentropy คือการลดทิศทางการ Error 
# Fitting ANN to the training set

classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 10)
classifier.summary()
# Predicting the Test set results

y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)

# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
cm