import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
titanic_train=pd.read_csv("../input/train.csv")

titanic_test=pd.read_csv("../input/train.csv")

knn = KNeighborsClassifier()
titanic_train.head()
titanic_test.head()
titanic_train.info()
titanic_test.info()
titanic_train.isnull().any()
Nan_List_train = [x for x in titanic_train.columns[titanic_train.isnull().any()] ]

print(Nan_List_train)

# using the isnull().any to check columns that contains any NAN, then get columns name into a list

for x in Nan_List_train:

    print(x,titanic_train[x].dtype)

    if(titanic_train[x].dtype == 'float64' or titanic_train[x].dtype == 'int64'):

        titanic_train[x].fillna(titanic_train[x].median(),inplace=True)

    if(titanic_train[x].dtype == 'object'):

        titanic_train[x].fillna('Nothing',inplace=True)

print(titanic_train.isnull().any())

# get columns name and type.

# Age is int

# Cabin is an Alpha and some number

# Embarked is is an Alpha

# hence fill float and int with colums median and object with string of 'Nothing'
Nan_List_test = [x for x in titanic_test.columns[titanic_test.isnull().any()] ]

print(Nan_List_test)

# using the isnull().any to check columns that contains any NAN, then get columns name into a list

for x in Nan_List_test:

    print(x,titanic_test[x].dtype)

    if(titanic_test[x].dtype == 'float64' or titanic_test[x].dtype == 'int64'):

        titanic_test[x].fillna(titanic_test[x].median(),inplace=True)

    if(titanic_test[x].dtype == 'object'):

        titanic_test[x].fillna('Nothing',inplace=True)

print(titanic_test.isnull().any())
titanic_train['FamilySize'] = titanic_train['SibSp'] + titanic_train['Parch'] + 1

# print(titanic_train['FamilySize'] )

# titanic_train['FamilySize']

# titanic_train.loc[titanic_train['FamilySize'] == 1]

titanic_test['FamilySize'] = titanic_test['SibSp'] + titanic_test['Parch'] + 1

titanic_train.drop(['PassengerId','Name','Ticket','Cabin','SibSp','Parch'],axis=1,inplace=True)

titanic_test.drop(['PassengerId','Name','Ticket','Cabin','SibSp','Parch'],axis=1,inplace=True)

# dropping irrelavent informations
# Parsing string values to int64 values.

for x in titanic_train:

    if (titanic_train[x].dtype != 'int64' and titanic_train[x].dtype != 'float64'):

        un = np.unique(titanic_train[x]).tolist()

        un.sort()

        for y in un:

            print( un.index(y))

            titanic_train[x] = titanic_train[x].replace(y, un.index(y)) 

for x in titanic_test:

    if (titanic_test[x].dtype != 'int64' and titanic_test[x].dtype != 'float64'):

        un = np.unique(titanic_test[x]).tolist()

        un.sort()

        for y in un:

            print( un.index(y))

            titanic_test[x] = titanic_test[x].replace(y, un.index(y)) 

# print(titanic_train[:])

# print(titanic_test[:])

x_train = titanic_train [[x for x in titanic_train.columns if x != 'Survived']]

x_train

y_train = titanic_train ['Survived']

y_train

x_test = titanic_train [[x for x in titanic_test.columns if x != 'Survived']]

x_test

y_test = titanic_test['Survived']

knn.fit(x_train, y_train) 

match = knn.predict(x_test)

accurate = [i for i, j in zip(match, y_test) if i == j]

print(len(accurate)/ y_test.size)