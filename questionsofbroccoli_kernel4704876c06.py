# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.


titanic_data = pd.read_csv('../input/train.csv', quotechar='"')

titanic_data.head()
titanic_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], 'columns', inplace=True)

titanic_data.head()
from sklearn import preprocessing



le = preprocessing.LabelEncoder()

titanic_data['Sex'] = le.fit_transform(titanic_data['Sex'].astype(str))

titanic_data.head()
titanic_data = pd.get_dummies(titanic_data, columns=['Embarked'])

titanic_data.head()
#titanic_data["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in titanic_data['Cabin'] ])
#titanic_data.head()
index_NaN_age = list(titanic_data["Age"][titanic_data["Age"].isnull()].index)



for i in index_NaN_age :

    age_med = titanic_data["Age"].median()

    age_pred = titanic_data["Age"][((titanic_data['SibSp'] == titanic_data.iloc[i]["SibSp"]) & (titanic_data['Parch'] == titanic_data.iloc[i]["Parch"]) & (titanic_data['Pclass'] == titanic_data.iloc[i]["Pclass"]))].median()

    if not np.isnan(age_pred) :

        titanic_data['Age'].iloc[i] = age_pred

    else :

        titanic_data['Age'].iloc[i] = age_med
titanic_data.describe()
titanic_data[titanic_data.isnull().any(axis=1)]
#from sklearn.model_selection import train_test_split



X = titanic_data.drop('Survived', axis=1)



Y = titanic_data['Survived']

from sklearn.ensemble import GradientBoostingRegressor



params = {'n_estimators': 500, 'max_depth': 6, 'min_samples_split': 2,

          'learning_rate': 0.01, 'loss': 'ls'}

gbr_model = GradientBoostingRegressor(**params)

gbr_model.fit(X, Y)
titanic_test = pd.read_csv('../input/test.csv', quotechar='"')

titanic_test.head()
test_IDs = titanic_test['PassengerId']

titanic_test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], 'columns', inplace=True)

titanic_test.head()
from sklearn import preprocessing



le = preprocessing.LabelEncoder()

titanic_test['Sex'] = le.fit_transform(titanic_test['Sex'].astype(str))

titanic_test.head()
titanic_test = pd.get_dummies(titanic_test, columns=['Embarked'])

titanic_test.head()

index_NaN_age = list(titanic_test["Age"][titanic_test["Age"].isnull()].index)



for i in index_NaN_age :

    age_med = titanic_test["Age"].median()

    age_pred = titanic_test["Age"][((titanic_test['SibSp'] == titanic_test.iloc[i]["SibSp"]) & (titanic_test['Parch'] == titanic_test.iloc[i]["Parch"]) & (titanic_test['Pclass'] == titanic_test.iloc[i]["Pclass"]))].median()

    if not np.isnan(age_pred) :

        titanic_test['Age'].iloc[i] = age_pred

    else :

        titanic_test['Age'].iloc[i] = age_med
titanic_test.head()
titanic_test[titanic_test.isnull().any(axis=1)]
titanic_test['Fare'].fillna((titanic_test['Fare'].mean()), inplace=True)

titanic_test[titanic_test.isnull().any(axis=1)]
test_y_predict = gbr_model.predict(titanic_test).round(0)

test_y_predict
survived = pd.Series(test_y_predict)

results = pd.concat([test_IDs, survived], axis=1)

results
results.to_csv("titanic_predict.csv",index=False)