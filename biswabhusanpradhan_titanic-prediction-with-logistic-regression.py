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
#Importing the datasets

train_origin = pd.read_csv('../input/titanic/train.csv')

test_origin = pd.read_csv('../input/titanic/test.csv')



#Extracting the Passenger Id from test to another variable for future reference

passenger_id = test_origin['PassengerId']



#Casting the target column as Categorical Data

y = train_origin['Survived'].astype('category')

train_origin = train_origin.drop(['Survived'], axis = 1)



#Combining both train and test datasets into one common dataset

dataset = pd.concat([train_origin, test_origin], axis=0, sort=False)



#Filling the missing values using feature engineering

dataset['Age'] = dataset.Age.fillna(dataset['Age'].median())

dataset['Cabin'] = dataset.Cabin.fillna('H0')

dataset.Embarked = dataset.Embarked.fillna('S').astype('category')

dataset.Pclass = dataset.Pclass.astype('category')

dataset.Sex = dataset.Sex.astype('category')

dataset.Fare = dataset.Fare.fillna(dataset['Fare'].median())

dataset.info()



#Introducing new feature Title using Feature Engineering

dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand = False)

dataset['Title'] = dataset['Title'].replace(['Lady','Countess','Capt','Col','Don','Dr', 'Major','Rev','Sir','Jonkheer','Dona'], 'Rare')

dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

dataset['Title'] = dataset['Title'].fillna('Unknown')

dataset['Title'] = dataset['Title'].astype('category')



#Labelling gender as binary variable. You can also use Label Encoding for the same

gender = {'male' : 0, 'female' : 1}

dataset['Sex'] = dataset['Sex'].map(gender)



#Creating another feature Family from Parch and SibSp which will store the number of family members

dataset['Family'] = dataset['Parch'] + dataset['SibSp']



#Creating Feature Deck from the first letter of Cabin

dataset['Deck'] = dataset['Cabin'].map(lambda x : x[0])



#Importing Label Encoder

from sklearn.preprocessing import LabelEncoder



#Encoding Deck, Title and Embarked column

dle = LabelEncoder()

dataset['Deck'] = dle.fit_transform(dataset['Deck'])

tle = LabelEncoder()

dataset['Title'] = tle.fit_transform(dataset['Title'])

ele = LabelEncoder()

dataset['Embarked'] = ele.fit_transform(dataset['Embarked'])



#Removing features which will no longer be used in our model builing process

dataset = dataset.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Cabin', 'Ticket'], axis = 1)
#scaling the features 

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(dataset)

dataset = scaler.transform(dataset)

print(dataset);
#Splitting the train and test set

from sklearn.model_selection import train_test_split

train_origin = dataset[0:891]

test_origin = dataset[891:1310]



#Training the Logistic Regression Model

from sklearn.linear_model import LogisticRegression

train_origin = dataset[0:891]

test_origin = dataset[891:1310]

X_train, X_test, y_train, y_test = train_test_split(train_origin, y)

lr_model = LogisticRegression()

lr_model.fit(X_train, y_train)

y_pred = lr_model.predict(test_origin)

lr_model.score(X_train, y_train)

lr_model.score(X_test, y_test)