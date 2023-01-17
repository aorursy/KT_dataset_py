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
data_train = pd.read_csv("../input/titanic/train.csv")
data_train.head()
data_test = pd.read_csv("../input/titanic/test.csv")
X_train = data_train.drop('Survived', axis=1)

y_train = data_train['Survived']

X = X_train.append(data_test, ignore_index=True)
X
X.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
X
X.replace(to_replace=['male', 'female'], value=[1, 0], inplace=True)
pd.unique(X['Embarked']).tolist()
X['Embarked'].replace(to_replace=['S', 'C', 'Q', np.nan], value=[1, 2, 3, 0], inplace=True)

pd.unique(X['Embarked']).tolist()
X['Cabin'].value_counts()
import math

Floors = []

for i in X['Cabin']:

    if type(i) == str:

        Floors.append(i[0])

    else:

        Floors.append(0)

X['Cabin'] = Floors
clas = pd.unique(X['Cabin']).tolist()

X['Cabin'].replace(to_replace=clas, value=[i for i in range(len(clas))], inplace=True)

X.fillna(0, inplace=True)

X
X.corr()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
print(np.sum(y_train == 1))

print(np.sum(y_train == 0))
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

cross_val_score(LogisticRegression(class_weight='balanced'), X_scaled[:len(y_train)], y_train, cv=3)
from sklearn.model_selection import GridSearchCV

LogisticRegression().get_params()
params = {'C': [0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],

         'class_weight': ['balanced', dict]}

grid_cv_mean = GridSearchCV(LogisticRegression(), params, cv = 3, scoring='accuracy')
grid_cv_mean.fit(X_scaled[:len(y_train)], y_train)

grid_cv_mean.best_params_
cross_val_score(LogisticRegression(C=0.01), X_scaled[:len(y_train)], y_train, cv=3, scoring='accuracy').mean()
from sklearn.preprocessing import PolynomialFeatures

transform = PolynomialFeatures(2)

X_transform = transform.fit_transform(X)

cross_val_score(LogisticRegression(C=0.01), X_transform[:len(y_train)], y_train, cv=3, scoring='accuracy').mean()
from sklearn.ensemble import RandomForestClassifier

cross_val_score(RandomForestClassifier(max_features='sqrt'),  X_scaled[:len(y_train)], y_train, cv=3, scoring='accuracy').mean()
from xgboost import XGBClassifier

cross_val_score(XGBClassifier(), X_scaled[:len(y_train)], y_train, cv=3, scoring='accuracy').mean()
best_classifier = LogisticRegression(C=0.01)

best_classifier.fit(X_transform[:len(y_train)], y_train)

predict = best_classifier.predict(X_transform[len(y_train):])
len(predict)
index = [i for i in range(892, 892 + len(predict))]

d = {'PassengerId': index, 'Survived': predict}

output = pd.DataFrame(d)
output
output.to_csv("../working/submission.csv",index=False)