# Importing libraries to play with data

import numpy as np

import pandas as pd
# Importing the data itself

train_set = pd.read_csv('../input/train.csv')

test_set = pd.read_csv('../input/test.csv')
train_set.info()
test_set.info()
train_set.head()
test_set.head()
# Importing matlab to plot graphs

import matplotlib as plt

%matplotlib inline
train_set.groupby('Sex').Survived.mean().plot(kind='bar')
train_set.groupby('Age').Survived.mean().plot(kind='line')
# Transforming the Sex into 0 and 1

train_set['Sex'] = train_set['Sex'].map({'male': 0, 'female': 1}).astype(int)
# Rounding the Age

train_set['Age'] = train_set.Age.round()
# Separating the data to predict the missing ages

X_train = train_set[train_set.Age.notnull()][['Pclass','Sex','SibSp','Parch', 'Fare']]

X_test = train_set[train_set.Age.isnull()][['Pclass','Sex','SibSp','Parch', 'Fare']]

y = train_set.Age.dropna()
# Predicting the missing ages

from sklearn.svm import SVC



age_classifier = SVC()

age_classifier.fit(X_train, y)

prediction = age_classifier.predict(X_test)

agePrediction = pd.DataFrame(data=prediction,index=X_test.index.values,columns=['Age'])

train_set = train_set.combine_first(agePrediction)
# Just confirming if there is no more ages missing

train_set.Age.isnull().sum()
from sklearn.model_selection import train_test_split



# Taking only the features that is important for now

X = train_set[['Sex', 'Age']]



# Taking the labels (Survived or Not Survived)

Y = train_set['Survived']



# Spliting into 80% for training set and 20% for testing set so we can see our accuracy

X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
X_train.info()
X_train.head()
len(Y_train)
Y_train.head()
# Importing C-Support Vector Classification from scikit-learn

from sklearn.svm import SVC



# Declaring the SVC with no tunning

classifier = SVC()



# Fitting the data. This is where the SVM will learn

classifier.fit(X_train, Y_train)



# Predicting the result and giving the accuracy

score = classifier.score(x_test, y_test)



print(score)
train_set.groupby('Pclass').Survived.mean().plot(kind='bar')
train_set.query('Sex == 1').groupby('Pclass').Survived.mean().plot(kind='bar')
# Taking only the features that is important for now

X = train_set[['Sex', 'Age', 'Pclass']]



# Taking the labels (Survived or Not Survived)

Y = train_set['Survived']



# Spliting into 80% for training set and 20% for testing set so we can see our accuracy

X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
# Declaring the SVC with no tunning

classifier = SVC()



# Fitting the data. This is where the SVM will learn

classifier.fit(X_train, Y_train)



# Predicting the result and giving the accuracy

score = classifier.score(x_test, y_test)



print(score)