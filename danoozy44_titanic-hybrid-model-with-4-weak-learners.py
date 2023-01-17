# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")
train = train.drop(columns=['Name','Cabin','Ticket'])

test = test.drop(columns=['Name','Cabin','Ticket'])
train['Embarked_S'] = (train['Embarked'] == 'S').astype(int)

train['Embarked_C'] = (train['Embarked'] == 'C').astype(int)

train['Embarked_Q'] = (train['Embarked'] == 'Q').astype(int)

train['Gender'] = (train['Sex'] == 'male').astype(int)
test['Embarked_S'] = (test['Embarked'] == 'S').astype(int)

test['Embarked_C'] = (test['Embarked'] == 'C').astype(int)

test['Embarked_Q'] = (test['Embarked'] == 'Q').astype(int)

test['Gender'] = (test['Sex'] == 'male').astype(int)
train = train.drop(columns = ['Sex'])

test = test.drop(columns = ['Sex'])



train = train.drop(columns = ['Embarked'])

test = test.drop(columns = ['Embarked'])
train.fillna(0, inplace=True)

test.fillna(0, inplace=True)
X = train.drop(columns=['Survived'])

y = train['Survived']
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
from lightgbm import LGBMClassifier

from xgboost import XGBClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import RandomForestClassifier



model_1 = LGBMClassifier(learning_rate=0.01, n_estimators=1000, max_depth=None)

model_2 = XGBClassifier(learning_rate=0.01, n_estimators=1000, max_depth=None)

model_3 = AdaBoostClassifier(learning_rate=0.01, n_estimators=1000)

model_4 = RandomForestClassifier(n_estimators=1000, random_state=42)
estimators = []



estimators.append(('lgbm',model_1))

estimators.append(('xgb',model_2))

estimators.append(('adaboost',model_3))

estimators.append(('clf',model_4))
from sklearn.ensemble import StackingClassifier



hybrid_model = StackingClassifier(estimators)
hybrid_model.fit(X_train, y_train)
pred = hybrid_model.predict(X_test)
from sklearn.metrics import accuracy_score



accuracy_score(pred, y_test)
test['Survived'] = actual_pred = hybrid_model.predict(test)
test[['PassengerId','Survived']].to_csv('submission.csv', index=False)