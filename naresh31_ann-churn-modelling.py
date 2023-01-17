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
base_dataset = pd.read_csv("../input/Churn_Modelling.csv")
base_dataset.head()
len(base_dataset.describe().columns)
len(base_dataset.describe(include='object').columns)
for i in base_dataset.describe(include='object').columns:

    from sklearn.preprocessing import LabelEncoder

    le=LabelEncoder()

    le.fit(base_dataset[i])

    x=le.transform(base_dataset[i])

    base_dataset[i]=x
base_dataset.head()
base_dataset.drop(['RowNumber','CustomerId'],axis = 1, inplace=True)
y=base_dataset['Exited']

x=base_dataset.drop('Exited',axis=1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(X_train)

X_train= pd.DataFrame(X_train)

X_test = sc.fit_transform(X_test)

X_test= pd.DataFrame(X_test)
# Importing the Keras libraries and packages

import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout
# Initialising the ANN

classifier = Sequential()
# Adding the input layer and the first hidden layer

classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# classifier.add(Dropout(p = 0.1))
# Adding the second hidden layer

classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# classifier.add(Dropout(p = 0.1))
# Adding the output layer

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fitting the ANN to the Training set

classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)