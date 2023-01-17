# Data analysis

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# data vsualization libraries

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.preprocessing import StandardScaler 

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier 

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

#from sklearn.metrics import confusion_matrix, accuracy_score, classification_report



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# loading datasets

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
X_test = test.copy()
train.head()
X_test.head()
train.info()
X_test.info()
train.describe()
X_test.describe()
train.describe(include=['O'])
sns.heatmap(train.corr(), cmap='coolwarm')

# correlation between all the features and lable
sns.heatmap(train.isnull())

# null values in dataset
train['Cabin'].value_counts().sum()

# Cabin feature contains only 204 non_null values
sns.countplot(x='Sex', data= train)

#count for male and female
sns.countplot(x='Sex', data=train, hue='Survived')

# out of total males and females how many are survived
train['Sex'].value_counts()
(577/891)*100

# 65% males are survived
(314/891)*100

# 35% females are survived
train['Sex'].groupby(train['Pclass']).value_counts().unstack()
train['Embarked'].value_counts()
train['Ticket'].value_counts()

# Ticket feature has duplicate values
train.drop(['PassengerId','Ticket','Name','Fare','Cabin'], axis=1, inplace=True)

#Fare is not correlated with Survived, so we dropped Fare

# PassengerId is not needed 

# Name is also not needed

# We dropped ticket because is contains many duplicate values

# Cabin contains so many missing values
train.head()
newdf_sex = pd.get_dummies(train['Sex'], drop_first=True)

newdf_Embarked = pd.get_dummies(train['Embarked'], drop_first=True)

train = pd.concat([train, newdf_sex, newdf_Embarked], axis=1)
train.drop(['Sex','Embarked'], axis=1, inplace=True)
train.head()
train[train['Pclass']==1]['Age'].mean()
train[train['Pclass']==2]['Age'].mean()
train[train['Pclass']==3]['Age'].mean()
def impute_age(cols):

    Age=cols[0]

    Pclass=cols[1]

    

    if pd.isnull(Age):

        if Pclass==1:

            return 38

        elif Pclass==2:

            return 29

        else:

            return 25

    else:

        return Age
train['Age'] = train[['Age','Pclass']].apply(impute_age, axis=1)
train.head()
train.info()
X_train = train.iloc[:,1:]

y_train = train['Survived']
X_test.info()
X_test.drop(['PassengerId','Name','Cabin','Fare'], axis=1, inplace=True)
newdf_sex = pd.get_dummies(X_test['Sex'], drop_first=True)

newdf_Embarked = pd.get_dummies(X_test['Embarked'], drop_first=True)

X_test = pd.concat([X_test, newdf_sex, newdf_Embarked], axis=1)
X_test.drop(['Sex','Embarked'], axis=1, inplace=True)
X_test[X_test['Pclass']==1]['Age'].mean()
X_test[X_test['Pclass']==2]['Age'].mean()
X_test[X_test['Pclass']==3]['Age'].mean()
def impute_age(cols):

    Age=cols[0]

    Pclass=cols[1]

    

    if pd.isnull(Age):

        if Pclass==1:

            return 41

        elif Pclass==2:

            return 29

        else:

            return 24

    else:

        return Age
X_test['Age'] = X_test[['Age','Pclass']].apply(impute_age, axis=1)
X_test.head()
X_test.drop('Ticket', axis=1, inplace=True)
X_test.head()
scaler = StandardScaler()

scaler.fit(X_train)

X_data = scaler.transform(X_train)
classifier = DecisionTreeClassifier()

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_pred
accuracy_decision_tree = round(classifier.score(X_train, y_train) * 100, 2)

accuracy_decision_tree
# random forest

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, y_train)

y_pred = random_forest.predict(X_test)

random_forest.score(X_train, y_train)

acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)

acc_random_forest
# Logistic regression

logistic_regression = LogisticRegression()

logistic_regression.fit(X_train, y_train)

y_pred = logistic_regression.predict(X_test)

logistic_regression.score(X_train, y_train)

acc_logistic_regression = round(logistic_regression.score(X_train,y_train)*100,2)

acc_logistic_regression
models = pd.DataFrame({

    'Model': ['Random Forest', 'Decision Tree','Logistic Regression'],

    'Score': [acc_random_forest, accuracy_decision_tree,acc_logistic_regression]})

models.sort_values(by='Score', ascending=False)
submission = pd.DataFrame({

     "PassengerId": test["PassengerId"],

        "Survived": y_pred

})
submission.to_csv('submission.csv', index=False)