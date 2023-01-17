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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/churn-modelling/Churn_Modelling.csv')
df.shape
df.head()
df.info()
df.describe()
X=df.loc[:,['CreditScore', 'Age', 'Tenure', 'Balance' , 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']].values

y=df.iloc[:,-1].values
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.25,random_state =0)
from sklearn.preprocessing import StandardScaler

sc_X =StandardScaler()

X_train =sc_X.fit_transform(X_train)

X_test =sc_X.transform(X_test)
from sklearn.svm import SVC

classifier = SVC(kernel = 'poly', degree = 2, random_state = 0) #degree for non-linear

classifier.fit(X_train, y_train) 
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)
X = df.iloc[:, 3:13].values

y = df.iloc[:, 13].values
print(X.shape)
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder



transformer = ColumnTransformer(

    transformers=[

        ("OneHot",

         OneHotEncoder(),

         [1,2]

        )

    ]

)



X2 = transformer.fit_transform(X)

print(X2.shape)

print(X2[0:20, :])
print(X2.shape)

print(X2[0:10, :])
X = np.concatenate((X[:,0:1],X[:,3:10], X2[:,1:4]), axis=1)

print(X.shape)

print(X[0:5, :])
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
classifier.summary()