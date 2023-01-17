# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import xgboost as xgb

from sklearn import model_selection

from sklearn.preprocessing import LabelEncoder

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/titanic/train.csv')

train.head()
train['Title'] = train['Name'].str.extract('([A-Za-z]+)\.', expand=False)

train.head()
meanAge = train.mean(axis=0,skipna=True)['Age']

meanAge
train['Age'].fillna(meanAge,inplace=True)

train['AgeBand'] = pd.cut(train['Age'], 5, labels=range(5)).astype(int)

train.head()
train['Title'] = train['Title'].replace(['Don', 'Capt', 'Col', 'Major', 'Sir', 'Jonkheer', 'Rev', 'Dr'], 'Honored')

train['Title'] = train['Title'].replace(['Lady', 'Dona', 'Mme', 'Countess'], 'Mrs')

train['Title'] = train['Title'].replace(['Mlle', 'Ms'], 'Miss')
train['Fare'] = train['Fare'].fillna(train.mean(axis=0,skipna=True)['Fare'])

train['Embarked'] = train['Embarked'].fillna('S')

train['CabinCode'] = (train['Cabin']

                        .str.slice(0,1)

                        .map({

                            'C':1, 

                            'E':2, 

                            'G':3,

                            'D':4, 

                            'A':5, 

                            'B':6, 

                            'F':7, 

                          

                        })

                        .fillna(0)

                        .astype(int))
train['Sex'] = LabelEncoder().fit_transform(train['Sex'])

train['TitleCode'] = LabelEncoder().fit_transform(train['Title'])

train['EmbarkedCode'] = LabelEncoder().fit_transform(train['Embarked'])

train.head()
params=[    'Pclass',

    'Sex',

    'Age',

    'SibSp',

    'Parch',

    'Fare',

    'TitleCode',

    'EmbarkedCode',

    'AgeBand',

    'CabinCode',

]

x_train = train[params]

y_train = train['Survived'].astype(int)

x_train.head()
parameters = {

    'n_estimators':[280,320],

    'max_depth':[4,5,6,7,8,9,10,11,12],

    'gamma':[1,2,3],

    #'max_delta_step':[0,1,2],

    'min_child_weight':[1,2,3], 

    #'colsample_bytree':[0.55,0.65],

    'learning_rate':[0.1,0.2,0.3],

    'subsample':[1,0.9,0.8],

    'base_score':[0.5]

}



grid = model_selection.GridSearchCV(xgb.XGBClassifier(), parameters, cv=5)

grid.fit(x_train, y_train)

print(grid.best_score_)



xg_boost = grid.best_estimator_

xg_boost
xg_boost.fit(x_train, y_train)

print(xg_boost.score(x_train, y_train))
