# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
"""

Created on Tue Oct 10 18:26:32 2017



@author: SRIJANPABBI

"""
dataset_test = pd.read_csv('../input/test.csv')

dataset_train = pd.read_csv('../input/train.csv')

dataset_train.head()

dataset_train.info()
dataset_train = dataset_train.drop(['PassengerId','Name','Ticket','Cabin','Fare'], axis = 1)

dataset_test= dataset_test.drop(['PassengerId','Name','Ticket','Cabin','Fare'], axis = 1)
# Taking care of missing data

# only in titanic_df, fill the two missing values with the most occurred value, which is "S".

dataset_train["Embarked"] = dataset_train["Embarked"].fillna("S")

dataset_train["Age"] = dataset_train["Age"].fillna(30)

dataset_test["Embarked"] = dataset_test["Embarked"].fillna("S")

dataset_test["Age"] = dataset_test["Age"].fillna(30)

dataset_test["Pclass"] = dataset_test["Pclass"].fillna(3)
X_train = dataset_train.iloc[:,1:].values

y_train = dataset_train.iloc[:,0].values

X_test = dataset_test.values
#NO need for so many label encoders(can work with one) 

#but for the sake of understanding the process easily

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

le1 = LabelEncoder()

le2 = LabelEncoder()

X_train[:,0] = le1.fit_transform(X_train[:,0])

X_test[:,0] = le2.fit_transform(X_test[:,0])

le3 = LabelEncoder()

le4 = LabelEncoder()

X_train[:,1] = le3.fit_transform(X_train[:,1])

X_test[:,1] = le4.fit_transform(X_test[:,1])

le5 = LabelEncoder()

le6 = LabelEncoder()

X_train[:,-1] = le5.fit_transform(X_train[:,-1])

X_test[:,-1] = le6.fit_transform(X_test[:,-1])

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)

imputer = imputer.fit(X_test)

X_test = imputer.transform(X_test)
ohe1 = OneHotEncoder(categorical_features = [0,5])

X_train = ohe1.fit_transform(X_train).toarray()

ohe2 = OneHotEncoder(categorical_features = [0,5])

X_test = ohe2.fit_transform(X_test).toarray()

X_train = X_train[:,[1,2,4,5,6,7,8,9,]]

X_test = X_test[:,[1,2,4,5,6,7,8,9]]
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
# Making the ANN

#import the Keras libraries and packages

import keras

from keras.models import Sequential

from keras.layers import Dense
classifier = Sequential()

#adding the i/p layer and the first hidden layer

# rectifier activation function by 'relu'

classifier.add(Dense(units=10,kernel_initializer='uniform', activation='relu',input_dim=8))

# adding secind layer

classifier.add(Dense(units=10,kernel_initializer='uniform', activation='relu'))

# adding final layer

classifier.add(Dense(units=1,kernel_initializer='uniform', activation='sigmoid'))
#Applying Stochastic gradient descent and compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#Training the ANN model

classifier.fit(X_train, y_train, batch_size=10, epochs=100)
# Predicting the Test set results

y_pred1 = classifier.predict(X_test)

y_pred = (y_pred1>0.5)

final = y_pred.astype(int)

final = np.array(final)

final = final.ravel()

dataset_test = pd.read_csv('../input/test.csv')

dataset_train = pd.read_csv('../input/train.csv')

d = {'PassengerId' : dataset_test.iloc[:,0].values, 'Survived':final}

dataframe = pd.DataFrame(data = d)

dataframe.to_csv("ANN_predictions.csv")
#assuming that the gender submissions file is ideal

df = pd.read_csv("../input/gender_submission.csv")

y_test = df.iloc[:,1].values
# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

cm