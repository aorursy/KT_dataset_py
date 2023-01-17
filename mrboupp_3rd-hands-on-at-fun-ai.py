import numpy as np

import pandas as pd



import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline
df = pd.read_csv('../input/titanic/train.csv')

train = df.drop(columns='PassengerId')

test = pd.read_csv('../input/titanic/test.csv')
train.head()
train.columns
test.columns
train.info()
train.describe(include='O')
train.isnull().sum()
train['Cabin'].fillna(train['Cabin'].mode()[0],inplace=True)

train['Embarked'].fillna(train['Embarked'].mode()[0],inplace=True)

train['Age'].fillna(train['Age'].mode()[0],inplace=True)
train.describe(include='O')
train.isnull().sum()
train.drop(columns=['Name','Ticket'],inplace=True)

train.info()
full_data = [train]

for train in full_data:

    train['FamilySize'] = train['SibSp'] + train['Parch'] + 1



train.drop(columns=['SibSp','Parch'],inplace=True)
train.info()
from sklearn import preprocessing

for column in ['Sex','Cabin','Embarked']:

    le = preprocessing.LabelEncoder()

    le.fit(train[column])

    train[column] = le.transform(train[column])



train.head()


sns.heatmap(train.corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)
sns.heatmap(train.corr())
from sklearn.model_selection import train_test_split

seed = 42

test_size = 0.22

y_train = train['Survived']

x_train = train.drop(['Survived'], axis=1)



x_train , x_test , y_train , y_test = train_test_split(x_train , y_train ,test_size = 0.1,random_state = 0)
test['Cabin'].fillna(test['Cabin'].mode()[0],inplace=True)

test['Embarked'].fillna(test['Embarked'].mode()[0],inplace=True)

test['Age'].fillna(test['Age'].mode()[0],inplace=True)

test.drop(columns=['Name','Ticket'],inplace=True)

test['Fare'].fillna(test['Fare'].mode()[0],inplace=True)



for column in ['Sex','Cabin','Embarked']:

    le = preprocessing.LabelEncoder()

    le.fit(test[column])

    test[column] = le.transform(test[column])
full_data = [test]

for test in full_data:

    test['FamilySize'] = test['SibSp'] + test['Parch'] + 1



test.drop(columns=['SibSp','Parch'],inplace=True)

Test=test.drop(columns='PassengerId')

Test.info()
from xgboost.sklearn import XGBClassifier

model = XGBClassifier(learning_rate=0.001,n_estimators=2500,

                                max_depth=4, min_child_weight=0,

                                gamma=0, subsample=0.7,

                                colsample_bytree=0.7,

                                scale_pos_weight=1, seed=27,

                                reg_alpha=0.00006)

model.fit(x_train, y_train)

pred = model.predict(Test)



submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": pred

    })

submission.to_csv('submission.csv', index=False)

print("done")