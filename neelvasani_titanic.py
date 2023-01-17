# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import matplotlib.pyplot as plt

'''import os

dirname, _, filenames in os.walk('/kaggle/input/titanic')'''

train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')



# Encoding categorical data





# extract title from name to judge whether male,female,married or unmarried

train_test_data = [train_data, test_data]

for data in train_test_data:

    data['Title'] = data['Name'].str.extract('([A-Za-z]+)\.', expand = False)



add_title = {"Mr": 0,"Miss": 1,"Mrs": 2,"Master": 3,"Dr": 3,"Rev": 3,"Mlle": 3,"Col": 3,"Major": 3,"Mme": 3,"Countess": 3,"Ms": 3,"Sir": 3,"Lady": 3,"Don": 3,"Capt": 3,"Jonkheer": 3,"Dona": 3}    

for data in train_test_data:

    data['Title'] = data['Title'].map(add_title)



# drop Name feature from train and test data

train_data.drop('Name',axis=1,inplace=True)

test_data.drop('Name',axis=1,inplace=True)

         



# handling missing values

train_data['Age'].fillna(train_data.groupby('Title')['Age'].transform('median'), inplace = True)

test_data['Age'].fillna(test_data.groupby('Title')['Age'].transform('median'), inplace = True)

for data in train_test_data:

    data.loc[data['Age'] <= 16, 'Age'] = 0,

    data.loc[(data['Age'] > 16) & (data['Age'] <= 26), 'Age'] = 1,

    data.loc[(data['Age'] > 26) & (data['Age'] <= 36), 'Age'] = 2,

    data.loc[(data['Age'] > 36) & (data['Age'] <= 62), 'Age'] = 3,

    data.loc[data['Age'] > 62, 'Age'] = 4



# Missing value of sex feature

add_sex = {"male": 0,"female": 1}

for data in train_test_data:

    data['Sex'] = data['Sex'].map(add_sex)   



# Missing value of Embarked feature

for data in train_test_data:

    data['Embarked'] = data['Embarked'].fillna('S')

add_embarked = {"S": 0,"C": 1,"Q": 2}

for data in train_test_data:

    data['Embarked'] = data['Embarked'].map(add_embarked)

    

# Handling missing value in fare

train_data['Fare'].fillna(train_data.groupby('Pclass')['Fare'].transform('median'),inplace = True)

test_data['Fare'].fillna(test_data.groupby('Pclass')['Fare'].transform('median'),inplace = True)

for data in train_test_data:

    data.loc[data['Fare'] <= 17, 'Fare'] = 0,

    data.loc[(data['Fare'] > 17) & (data['Fare'] <= 30), 'Fare'] = 1,

    data.loc[(data['Fare'] > 30) & (data['Fare'] <= 100), 'Fare'] = 2,

    data.loc[data['Fare'] > 100, 'Fare'] = 3

    

    

# cabin data

for data in train_test_data:

    data['Cabin'] = data['Cabin'].str[:1]

    

add_cabin = {"A": 0,"B": 0.4,"C": 0.8,"D": 1.2,"E": 1.6,"F": 2,"G": 2.4,"T": 2.8,}

for data in train_test_data:

    data['Cabin'] = data['Cabin'].map(add_cabin)



train_data['Cabin'].fillna(train_data.groupby('Pclass')['Cabin'].transform('median'),inplace = True)

test_data['Cabin'].fillna(test_data.groupby('Pclass')['Cabin'].transform('median'),inplace = True)



# new feature family size from siblings and parch

train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1

test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1



add_family = {1: 0,2: 0.4,3: 0.8,4: 1.2,5: 1.6,6: 2,7: 2.4,8: 2.8,9: 3.2,10: 3.6,11: 4,}

for data in train_test_data:

    data['FamilySize'] = data['FamilySize'].map(add_family)



# drop useless features

drop_feature = ['Ticket', 'SibSp', 'Parch']

train_data = train_data.drop(drop_feature, axis=1)

train_data = train_data.drop(['PassengerId'], axis = 1)

test_data = test_data.drop(drop_feature, axis=1)



# change train_data to X_train and X_test

X_test = pd.DataFrame({"Survived": train_data['Survived']})

X_train = train_data.drop('Survived', axis = 1)





# classifying model 

'''from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)'''

from sklearn import svm





# Testing SVM

clf = svm.SVC()

clf.fit(X_train, X_test)  

test = test_data.drop("PassengerId", axis = 1).copy()

prediction = clf.predict(test)



submission = pd.DataFrame({

    "PassengerId": test_data["PassengerId"],

    "Survived": prediction

})

submission.to_csv('submission.csv', index = False)

submission = pd.read_csv('submission.csv')

   

# Any results you write to the current directory are saved as output.
train_data.head()

test['Title'].value_counts()
train_data
test_data
X_train
X_test['Survived'].value_counts()
submission.head()
X_test
submission