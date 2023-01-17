# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split, RandomizedSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier



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
train.head()
train.columns
test.columns
y = train['Survived']

X = train.drop(['Survived'], axis=1)
eda = pd.concat([X, test])

len(eda)
eda.nunique(axis=0, dropna=True)
for column in eda.columns:

    print('{} has {} nulls'.format(column, eda[column].isna().sum()))
X = X.drop(['Name', 'Cabin'], axis=1)
X['Sex'] = LabelEncoder().fit_transform(X['Sex'])

X['Embarked'] = X['Embarked'].fillna('S')

X['Embarked'] = LabelEncoder().fit_transform(X['Embarked'])

X['Ticket'] = LabelEncoder().fit_transform(X['Ticket'])

X.fillna(0, inplace=True)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8)
rfc = RandomForestClassifier()

params = {'n_estimators':[10, 25, 50, 75, 100], 'max_depth':[4, 5, 6, 7, 8]}

model1 = RandomizedSearchCV(rfc, params).fit(X_train, y_train)

model1.best_params_, model1.score(X_valid, y_valid)
rbf = SVC()

params = {'C':[0.01, 0.1, 10, 100, 1000], 'gamma':[0.01, 0.1, 10, 100, 1000]}

model2 = RandomizedSearchCV(rbf, params).fit(X_train, y_train)

model2.best_params_, model2.score(X_valid, y_valid)
xgb = XGBClassifier(n_jobs=-1)

params = {'learning_rate':[0.1, 0.2, 0.08, 0.05], 'max_depth':[3, 4, 5, 6, 7], 'n_estimators':[100, 125, 150, 175]}

model3 = RandomizedSearchCV(xgb, params).fit(X_train, y_train)

model3.best_params_, model3.score(X_valid, y_valid)
test = test.drop(['Name', 'Cabin'], axis=1)

test['Sex'] = LabelEncoder().fit_transform(test['Sex'])

test['Embarked'] = X['Embarked'].fillna('S')

test['Embarked'] = LabelEncoder().fit_transform(test['Embarked'])

test['Ticket'] = LabelEncoder().fit_transform(test['Ticket'])

test.fillna(0, inplace=True)
test
prediction = model1.predict(test)
len(prediction), len(test)
df = pd.DataFrame()

df['PassengerId'] = test['PassengerId']

df['Survived'] = prediction

df.to_csv('submit.csv', index=False)