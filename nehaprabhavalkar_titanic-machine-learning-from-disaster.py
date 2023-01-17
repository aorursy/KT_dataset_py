# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np

import pandas as pd 

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

import xgboost as xb

from sklearn.linear_model import LogisticRegression

import warnings 

warnings.filterwarnings('ignore')

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



train_data=pd.read_csv('/kaggle/input/titanic/train.csv')

test_data=pd.read_csv('/kaggle/input/titanic/test.csv')
train_data.shape
train_data.describe()
train_data.columns
train_data.info()
train_data.isnull().sum()
test_data.isnull().sum()
sns.distplot(train_data['Age'],kde=True)
train_data['Age'] = train_data['Age'].fillna(30)
train_data = train_data.drop(['Cabin'],axis=1)

test_data = test_data.drop(['Cabin'],axis=1)
train_data['Embarked'].value_counts()
train_data['Embarked']=train_data['Embarked'].fillna('S')
train_data['Sex'] = pd.get_dummies(train_data['Sex'])

test_data['Sex'] = pd.get_dummies(test_data['Sex'])

train_data['Fare']=train_data['Fare'].astype('int32')

train_data['Embarked'] = pd.factorize(train_data['Embarked'])[0]

test_data['Embarked'] = pd.factorize(test_data['Embarked'])[0]

test_data['Fare']=test_data['Fare'].fillna(test_data['Fare'].mean())

test_data['Fare']=test_data['Fare'].astype('int32')
sns.heatmap(train_data.corr(),annot=True,linewidths=0.5)
sns.countplot(x=train_data['Pclass'],hue=train_data['Survived'])
sns.countplot(x=train_data['Embarked'],hue=train_data['Survived'])
sns.countplot(x=train_data['Age'],hue=train_data['Survived'])
for i in range(0,len(train_data)):

    if train_data['Age'][i] <= 15:

        train_data['Age'][i] = 0

    elif (train_data['Age'][i] > 15) & (train_data['Age'][i] <=35):

        train_data['Age'][i]=1

    elif (train_data['Age'][i] > 35) & (train_data['Age'][i] <=55):

        train_data['Age'][i]=2

    elif (train_data['Age'][i] > 55) & (train_data['Age'][i] <=75):

        train_data['Age'][i]=3

    else:

        train_data['Age'][i]=4
for i in range(0,len(test_data)):

    if test_data['Age'][i] <= 15:

        test_data['Age'][i] = 0

    elif (test_data['Age'][i] > 15) & (test_data['Age'][i] <=26):

        test_data['Age'][i]=1

    elif (test_data['Age'][i] > 35) & (test_data['Age'][i] <=55):

        test_data['Age'][i]=2

    elif (test_data['Age'][i] > 55) & (test_data['Age'][i] <=75):

        test_data['Age'][i]=3

    else:

        test_data['Age'][i]=4
sns.countplot(x=train_data['Fare'],hue=train_data['Pclass'])
for i in range(0,len(train_data)):

    if train_data['Fare'][i] <= 50:

        train_data['Fare'][i] = 3

    elif (train_data['Fare'][i] > 50) & (train_data['Fare'][i] <=150 ):

        train_data['Fare'][i]=2

    else:

        train_data['Fare'][i]=1
for i in range(0,len(test_data)):

    if test_data['Fare'][i] <= 50:

        test_data['Fare'][i] = 3

    elif (test_data['Fare'][i] > 50) & (test_data['Fare'][i] <=150 ):

        test_data['Fare'][i]=2

    else:

        test_data['Fare'][i]=1
train_data['Fam'] = train_data['Parch'] + train_data['SibSp']

test_data['Fam'] = test_data['Parch'] + test_data['SibSp']
sns.countplot(train_data['Fam'])
for i in range(0,len(train_data)):

    if train_data['Fam'][i] == 0:

        train_data['Fam'][i] = 0

    elif (train_data['Fam'][i] >= 1) & (train_data['Fam'][i] <=3):

        train_data['Fam'][i]=1

    elif (train_data['Fam'][i] >= 4) & (train_data['Fam'][i] <=6):

        train_data['Fam'][i]=2

    elif (train_data['Fam'][i] >= 7) & (train_data['Fam'][i] <=9):

        train_data['Fam'][i]=3

    else:

        train_data['Fam'][i]=4
for i in range(0,len(test_data)):

    if test_data['Fam'][i] == 0:

        test_data['Fam'][i] = 0

    elif (test_data['Fam'][i] >= 1) & (test_data['Fam'][i] <=3):

        test_data['Fam'][i]=1

    elif (test_data['Fam'][i] >= 4) & (test_data['Fam'][i] <=6):

        test_data['Fam'][i]=2

    elif (test_data['Fam'][i] >= 7) & (test_data['Fam'][i] <=9):

        test_data['Fam'][i]=3

    else:

        test_data['Fam'][i]=4
X = train_data[['Sex','Pclass','Age','Parch','Fam','Fare','Embarked']]

y = train_data[['Survived']]

X_test = test_data[['Sex','Pclass','Age','Parch','Fam','Fare','Embarked']]

train_X , test_X , train_y , test_y = train_test_split(X,y,test_size = 0.2,random_state=0)
classifier = RandomForestClassifier(n_estimators=500,max_depth=3)

classifier.fit(train_X,train_y)

classifier.score(test_X,test_y)
xgb_model = xb.XGBClassifier(base_score=0.5,n_estimators=1000, learning_rate=0.05)

xgb_model.fit(train_X, train_y, early_stopping_rounds=5, 

             eval_set=[(test_X, test_y)], verbose=False)

xgb_model.score(test_X,test_y)
log_reg = LogisticRegression()

log_reg.fit(train_X,train_y)

log_reg.score(test_X,test_y)
predictions = xgb_model.predict(X_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('submission.csv', index=False)