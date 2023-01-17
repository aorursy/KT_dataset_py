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
train = train.drop(['Cabin', 'Embarked', 'Name', 'Ticket', 'PassengerId'],axis=1)

test = test.drop(['Cabin', 'Embarked', 'Name', 'Ticket'],axis=1)
train["Age"].fillna(train.groupby("Sex")["Age"].transform("mean"), inplace=True)

test["Age"].fillna(test.groupby("Sex")["Age"].transform("mean"), inplace=True)

test["Fare"].fillna(test.groupby("Sex")["Fare"].transform("median"), inplace=True)
train.isnull().sum()

test.isnull().sum()
sex_mapping = {"male": 0, "female": 1}

train['Sex'] = train['Sex'].map(sex_mapping)

test['Sex'] = test['Sex'].map(sex_mapping)
age_mean = train['Age'].mean()

age_std = train['Age'].std()

indexNames = train[train['Age'] < age_mean - 3*age_std].index

train.drop(indexNames , inplace=True)

indexNames = train[train['Age'] > age_mean + 3*age_std].index

train.drop(indexNames , inplace=True)
fare_mean = train['Fare'].mean()

fare_std = train['Fare'].std()

indexNames = train[train['Fare'] < fare_mean - 3*fare_std].index

train.drop(indexNames , inplace=True)

indexNames = train[train['Fare'] > fare_mean + 3*fare_std].index

train.drop(indexNames , inplace=True)
from sklearn.linear_model import LogisticRegression

ml = LogisticRegression(solver='lbfgs')
x = train.drop(['Survived', 'Age', 'Parch', 'Fare', 'SibSp'], axis=1)

y = train['Survived']

ml.fit(x, y)
ml.coef_
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)



scoring = 'accuracy'

score = cross_val_score(ml, x, y, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)



round(np.mean(score)*100, 2)
predict = ml.predict(test.drop(['PassengerId', 'Age', 'Parch', 'Fare', 'SibSp'], axis=1))

result =pd.DataFrame({

    'PassengerId': test['PassengerId'],

    'Survived': predict

})

result.to_csv('result.csv', index=False) 