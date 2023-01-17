import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")
train.info()
train.shape
train.isnull().sum()
test.info()
test.shape
print("test data is "+str((test.shape[0]/(train.shape[0]+test.shape[0]))*100)+"% of total data")
train.describe()
train.head(10)
plt.hist(train['Fare'], bins=50)

plt.xlabel('number of passengers')

plt.ylabel('fare')

plt.title('Distribution of Fares')

plt.show()
train.groupby('Sex')['PassengerId'].count()
sns.countplot('Sex', data=train)

plt.show()
train.groupby(['Sex','Survived'])['Survived'].count()
sns.countplot('Sex', hue='Survived', data=train)

plt.title('Sex vs Survived')

plt.show()
def survival_bar(input):

    survived = train[train['Survived']==1][input].value_counts()

    dead = train[train['Survived']==0][input].value_counts()

    df = pd.DataFrame([survived, dead])

    df.index = ['Survived', 'Dead']

    df.plot(kind='bar', stacked=True)
survival_bar('Sex')
train.groupby('Pclass')['PassengerId'].count()
sns.countplot('Pclass', data=train)

plt.show()
train.groupby(['Pclass','Survived'])['Survived'].count()
sns.countplot('Pclass', hue='Survived', data=train)

plt.title('Pclass vs Survived')

plt.show()
survival_bar('Pclass')
sns.countplot('Embarked', data=train)

plt.show()
train['Embarked'] = train['Embarked'].fillna('S')
survival_bar('Embarked')
train.groupby(['Pclass','Embarked'])['Embarked'].count()
sns.countplot('Pclass', hue='Embarked', data=train)

plt.title('Pclass vs Embarked')

plt.show()
#train.loc[train['Embarked'] == 'S', 'Embarked'] = 0

#train.loc[train['Embarked'] == 'C', 'Embarked'] = 1

#train.loc[train['Embarked'] == 'Q', 'Embarked'] = 2



embarked_mapping = {"S": 0, "C": 1, "Q": 2}

train['Embarked'] = train['Embarked'].map(embarked_mapping)
corr_matrix = train.corr()

corr_matrix["Survived"].sort_values(ascending=False)
sex_mapping = {"male": 0, "female": 1}

train['Sex'] = train['Sex'].map(sex_mapping)
corr_matrix = train.corr()

corr_matrix["Survived"].sort_values(ascending=False)
survival_bar('SibSp')
survival_bar('Parch')
train.plot(kind="scatter", x="Age", y="Survived")
train.plot(kind="scatter", x="Age", y="Survived", figsize=(20,3))
train.loc[train['Age'] <= 10, 'Age'] = 0,

train.loc[(train['Age'] > 10) & (train['Age'] <= 20), 'Age'] = 1,

train.loc[(train['Age'] > 20) & (train['Age'] <= 40), 'Age'] = 2,

train.loc[(train['Age'] > 40) & (train['Age'] <= 60), 'Age'] = 3,

train.loc[train['Age'] > 60, 'Age'] = 4
survival_bar('Age')
corr_matrix = train.corr()

corr_matrix["Survived"].sort_values(ascending=False)
train.head(10)
train.info()
train['Title'] = train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
train['Title'].value_counts()
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 

                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,

                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }

train['Title'] = train['Title'].map(title_mapping)
train.info()
train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)
train.isnull().sum()
train_prepared = train.drop('Survived', axis=1)

train_label = train['Survived'].copy()
train_prepared = train_prepared.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
train_prepared.info()
train_prepared.head(10)
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier



rfc = RandomForestClassifier(n_estimators = 100,random_state = 42)

score = cross_val_score(rfc, train_prepared, train_label, cv=10, n_jobs=1, scoring='accuracy')

print(score)
round(np.mean(score)*100, 2)
test.isnull().sum()
test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)
test['Title'] = test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
test['Title'] = test['Title'].map(title_mapping)
test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)
test.isnull().sum()
test.loc[test['Age'] <= 10, 'Age'] = 0,

test.loc[(test['Age'] > 10) & (test['Age'] <= 20), 'Age'] = 1,

test.loc[(test['Age'] > 20) & (test['Age'] <= 40), 'Age'] = 2,

test.loc[(test['Age'] > 40) & (test['Age'] <= 60), 'Age'] = 3,

test.loc[test['Age'] > 60, 'Age'] = 4
sex_mapping = {"male": 0, "female": 1}

test['Sex'] = test['Sex'].map(sex_mapping)
embarked_mapping = {"S": 0, "C": 1, "Q": 2}

test['Embarked'] = test['Embarked'].map(embarked_mapping)
test['Embarked'] = test['Embarked'].fillna('S')
test_prepared = test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1).copy()
rfc = RandomForestClassifier(n_estimators = 100,random_state = 42)

rfc.fit(train_prepared, train_label)
prediction = rfc.predict(test_prepared)
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": prediction

    })



submission.to_csv('submission.csv', index=False)