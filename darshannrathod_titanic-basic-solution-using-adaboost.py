import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import os

print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")

train.head()
test = pd.read_csv("../input/test.csv")

test.head()
train.info()
test.info()
all = pd.concat([train, test], sort = False)

all.info()
#Fill Missing numbers with median

all['Age'] = all['Age'].fillna(value=all['Age'].median())

all['Fare'] = all['Fare'].fillna(value=all['Fare'].median())
all.info()
sns.catplot(x = 'Embarked', kind = 'count', data = all) #or all['Embarked'].value_counts()
all['Embarked'] = all['Embarked'].fillna('S')

all.info()
#Age

all.loc[ all['Age'] <= 16, 'Age'] = 0

all.loc[(all['Age'] > 16) & (all['Age'] <= 32), 'Age'] = 1

all.loc[(all['Age'] > 32) & (all['Age'] <= 48), 'Age'] = 2

all.loc[(all['Age'] > 48) & (all['Age'] <= 64), 'Age'] = 3

all.loc[ all['Age'] > 64, 'Age'] = 4 
#Title

import re

def get_title(name):

    title_search = re.search(' ([A-Za-z]+\.)', name)

    

    if title_search:

        return title_search.group(1)

    return ""
all['Title'] = all['Name'].apply(get_title)

all['Title'].value_counts()
all['Title'] = all['Title'].replace(['Capt.', 'Dr.', 'Major.', 'Rev.'], 'Officer.')

all['Title'] = all['Title'].replace(['Lady.', 'Countess.', 'Don.', 'Sir.', 'Jonkheer.', 'Dona.'], 'Royal.')

all['Title'] = all['Title'].replace(['Mlle.', 'Ms.'], 'Miss.')

all['Title'] = all['Title'].replace(['Mme.'], 'Mrs.')

all['Title'].value_counts()
#Cabin

all['Cabin'] = all['Cabin'].fillna('Missing')

all['Cabin'] = all['Cabin'].str[0]

all['Cabin'].value_counts()
#Family Size & Alone 

all['Family_Size'] = all['SibSp'] + all['Parch'] + 1

all['IsAlone'] = 0

all.loc[all['Family_Size']==1, 'IsAlone'] = 1

all.head()
#Drop unwanted variables

all_1 = all.drop(['Name', 'Ticket'], axis = 1)

all_1.head()
all_dummies = pd.get_dummies(all_1, drop_first = True)

all_dummies.head()
all_train = all_dummies[all_dummies['Survived'].notna()]

all_train.info()
all_test = all_dummies[all_dummies['Survived'].isna()]

all_test.info()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(all_train.drop(['PassengerId','Survived'],axis=1), 

                                                    all_train['Survived'], test_size=0.30, 

                                                    random_state=101, stratify = all_train['Survived'])
from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier
ada = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=100, random_state=0)

ada.fit(X_train,y_train)
predictions = ada.predict(X_test)
from sklearn.metrics import classification_report

print(classification_report(y_test,predictions))
print (f'Train Accuracy - : {ada.score(X_train,y_train):.3f}')

print (f'Test Accuracy - : {ada.score(X_test,y_test):.3f}')
TestForPred = all_test.drop(['PassengerId', 'Survived'], axis = 1)
t_pred = ada.predict(TestForPred).astype(int)
PassengerId = all_test['PassengerId']
adaSub = pd.DataFrame({'PassengerId': PassengerId, 'Survived':t_pred })

adaSub.head()
adaSub.to_csv("1_Ada_Submission.csv", index = False)