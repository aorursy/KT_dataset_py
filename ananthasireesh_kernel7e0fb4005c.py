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

dataset = pd.read_csv('/kaggle/input/predicting-churn-for-bank-customers/Churn_Modelling.csv')

print(dataset.info())

print(dataset.head())
X = dataset.iloc[:, 3:13].values

y = dataset.iloc[:, 13].values
# Encoding categorical data

# Encoding the Independent Variable

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()

X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1]) #country encoding

print(X[:, 1])

labelencoder_X_2 = LabelEncoder()

X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2]) #sex encoding

print(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1],handle_unknown='ignore')

X = onehotencoder.fit_transform(X).toarray() #create dummy variable for country

print(X[:,0:3])

X=X[:,1:] #leave out first column
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
#import keras libraries and packages

import keras 

from keras.models import Sequential

from keras.layers import Dense
#build the ANN

#intialising ANN

classifier=Sequential()



#Dense function is used for intialising weights 

#output_dim is number of nodes, init is how to intialise, activation function is the optmiser, input_dim is number of nodes

#input layer

#number of input layer nodes is no of independent variables 

classifier.add(Dense(output_dim=6,init="uniform",activation='relu',input_dim=11))

classifier.add(Dense(output_dim=6,init="uniform",activation='relu'))

classifier.add(Dense(output_dim=1,init="uniform",activation='sigmoid'))
#compile whole ANN model applying shocastic gradeient optimiser

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#fitting ANN to data 

classifier.fit(X_train,y_train,epochs=100,batch_size=10)
# Predicting the Test set results

y_pred = classifier.predict(X_test)

y_pred=(y_pred>0.5)

# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)
# Predicting a single new observation

"""Predict if the customer with the following informations will leave the bank:

Geography: France

Credit Score: 600

Gender: Male

Age: 40

Tenure: 3

Balance: 60000

Number of Products: 2

Has Credit Card: Yes

Is Active Member: Yes

Estimated Salary: 50000"""

new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))

new_prediction = (new_prediction > 0.5)



print(new_prediction)