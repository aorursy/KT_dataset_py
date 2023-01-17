# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from xgboost import XGBClassifier

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data
#test_data_with_labels = pd.read_csv('/kaggle/input/titanic/titanic.csv')
test_data['Sex']


test_data.loc[test_data['Sex'] == "male", 'Sex'] = 1

test_data.loc[test_data['Sex'] == "female", 'Sex'] = 0



train_data.loc[train_data['Sex'] == "female", 'Sex'] = 0

train_data.loc[train_data['Sex'] == "male", 'Sex'] = 1
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 

                              GradientBoostingClassifier, ExtraTreesClassifier)













# Put in our parameters for said classifiers

# Random Forest parameters

rf_params = {

    'n_jobs': -1,

    'n_estimators': 500,

     'warm_start': True, 

     #'max_features': 0.2,

    'max_depth': 6,

    'min_samples_leaf': 2,

    'max_features' : 'sqrt',

    'verbose': 0

}



# Extra Trees Parameters

et_params = {

    'n_jobs': -1,

    'n_estimators':500,

    #'max_features': 0.5,

    'max_depth': 8,

    'min_samples_leaf': 2,

    'verbose': 0

}



# AdaBoost parameters

ada_params = {

    'n_estimators': 500,

    'learning_rate' : 0.75

}



# Gradient Boosting parameters

gb_params = {

    'n_estimators': 500,

     #'max_features': 0.2,

    'max_depth': 5,

    'min_samples_leaf': 2,

    'verbose': 0

}



# Support Vector Classifier parameters 

svc_params = {

    'kernel' : 'linear',

    'C' : 0.025

    }







'''model = RandomForestClassifier(n_estimators=100, 

                               bootstrap = True,

                               max_features = 'sqrt')'''





"""rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)

et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)

ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)

gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)

svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)

"""
test_data.columns
model = XGBClassifier()

y = train_data["Survived"]





train_data['Age'] = train_data['Age'].fillna(0)

test_data['Age'] = test_data['Age'].fillna(0)

features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train_data[features])

#X = test_data.dropna()

X_test = pd.get_dummies(test_data[features])



X_test.loc[X_test['Sex'] == "female", 'Sex'] = 0

X_test.loc[X_test['Sex'] == "male", 'Sex'] = 1





model.fit(X, y)


model.score(X, y)
predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")