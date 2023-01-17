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
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
train.count()
test.count()
train.head()
test.head()
import seaborn as sns

import matplotlib.pyplot as plt

#Check the missing data using heatmap

sns.heatmap(train.isnull(), yticklabels=False, cbar = False, cmap = 'viridis')

plt.show()
sns.heatmap(test.isnull(), yticklabels=False, cbar = False, cmap = 'viridis')

plt.show()
train.isnull().sum()
test.isnull().sum()
combine = [train, test]

for dataset in combine:    

    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)

    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)

    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)

    

combine = [train,test]

test.isnull().sum()  
train.isnull().sum()
train.drop(['Cabin', 'Name','Ticket'],axis=1, inplace=True)

test.drop(['Cabin', 'Name','Ticket'],axis=1, inplace=True)

train.head()
test.head()
#Create a new variable sex with only one column, 1 as male and 0 as female.

sex = pd.get_dummies(train['Sex'],drop_first=True)

sex1 = pd.get_dummies(test['Sex'],drop_first=True)

#Create a new variable embark with two column, if both Q and S are zero it means C is 1. 

embark= pd.get_dummies(train['Embarked'],drop_first=True)

embark1= pd.get_dummies(test['Embarked'],drop_first=True)

#Drop the old column 'Sex' and 'Embarked'

train=train.drop(['Sex','Embarked','PassengerId'],axis=1)

test=test.drop(['Sex','Embarked'],axis=1)



#Create a new data set with quantitative information

train_new = pd.concat([train, sex, embark],axis=1)

test_new = pd.concat([test, sex1, embark1],axis=1)

train_new.head()
test_new.head()
#Create your input and ouput data set

X_train= train_new.drop('Survived',axis=1)

Y_train=train_new['Survived']

X_test  = test_new.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape



from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train, Y_train)

predictions= lr.predict(X_test)

acc_log = round(lr.score(X_train, Y_train) * 100, 2)

acc_log
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

prediction = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

acc_decision_tree

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest
Y_pred = random_forest.predict(X_test)

submission = pd.DataFrame({"PassengerId": test_new["PassengerId"],

                          "Survived": Y_pred})

submission
submission.to_csv('My_Submission.csv', index=False)

print("Your Submission was successfully saved!")