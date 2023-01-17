# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
train.head(5)
#Dummy variables

dummy = pd.get_dummies(train['Sex'])

train.pop('Sex')

train = pd.concat([train, dummy], axis=1)



dummy = pd.get_dummies(test['Sex'])

test.pop('Sex')

test = pd.concat([test, dummy], axis=1)
#Describe Train

train.describe()
#Describe Test

test.describe()
#values null

train.isnull().sum(),  test.isnull().sum()
train['Cabin'] = train['Cabin'].fillna(0)

train['Cabin'] = train['Cabin'].apply(lambda x: 0 if x==0 else 1)

test['Cabin'] = test['Cabin'].fillna(0)

test['Cabin'] = test['Cabin'].apply(lambda x: 0 if x==0 else 1)
#Remove missing values

train = train.dropna()

mean = test['Cabin'].mean()

test = test.fillna(mean)
train.head(5)
#Select features

x_label = ['Pclass', 'Age', 'SibSp', 'Parch', 'female', 'male', 'Cabin']

y_label = ['Survived']
#Split

X_train, X_validation, y_train, y_validation = train_test_split(train[x_label], train[y_label], test_size=0.33)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score
rfc = RandomForestClassifier(random_state=42)

param_grid = { 

    'n_estimators': [100, 200, 300, 400, 500],

    'max_features': ['auto', 'sqrt', 'log2'],

    'max_depth' : [4,5,6,7,8, 10],

    'criterion' :['gini', 'entropy']

}



gs_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv = 5)

gs_rfc.fit(X_train, y_train)
gs_rfc.best_params_
rfc_best = RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 100, max_depth=4, criterion='gini')

rfc_best.fit(X_train, y_train)
pred = rfc_best.predict(X_validation)

print(accuracy_score(y_validation, pred))
#Submit result

pred_test = rfc_best.predict(test[x_label])

result=pd.DataFrame(test['PassengerId'])

result['Survived']=pred_test

result.to_csv("op_rf.csv", index=False)