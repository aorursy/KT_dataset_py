!pip install -U interpret
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



np.random.seed(777)



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

data = pd.concat([train, test], sort=True)
data['Sex'].replace(['male','female'],[0, 1], inplace=True)

data['Embarked'].fillna(('S'), inplace=True)

data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

data['Fare'].fillna(np.mean(data['Fare']), inplace=True)

age_avg = data['Age'].mean()

age_std = data['Age'].std()

data['Age'].fillna(np.random.randint(age_avg - age_std, age_avg + age_std), inplace=True)

delete_columns = ['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin']

data.drop(delete_columns, axis = 1, inplace = True)
train = data[:len(train)]

test = data[len(train):]

y_train = train['Survived']

X_train = train.drop('Survived', axis = 1)

X_test = test.drop('Survived', axis = 1)

X_train = X_train.values

X_test = X_test.values
from interpret.glassbox import ExplainableBoostingClassifier



ebm = ExplainableBoostingClassifier(n_jobs=-1, interactions=0, random_state=777)

ebm.fit(X_train, y_train)



# EBM supports pandas dataframes, numpy arrays, and handles "string" data natively.
y_pred = ebm.predict(X_test)
sub = pd.DataFrame(pd.read_csv("../input/test.csv")['PassengerId'])

sub['Survived'] = list(map(int, y_pred))

sub.to_csv("submission.csv", index = False)
# from interpret import show



# ebm_global = ebm.explain_global()

# show(ebm_global)
# ebm_local = ebm.explain_local(X_test, y_test)

# show(ebm_local)
# show([logistic_regression, decision_tree])