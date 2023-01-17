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
# load train and test data

train_full = pd.read_csv('../input/titanic/train.csv')

test_full = pd.read_csv('../input/titanic/test.csv')



y = train_full['Survived']

train_full.drop(columns='Survived', inplace=True)



# modifying gender to 0|1

m = {'m' : 1, 'f' : 0}

train_full['Sex'] = train_full['Sex'].str[0].map(m)

test_full['Sex'] = test_full['Sex'].str[0].map(m)



# Cabin

def cabin(data):

    data.Cabin.fillna('0', inplace=True)

    data.loc[data.Cabin.str[0] == 'A', 'Cabin'] = 1

    data.loc[data.Cabin.str[0] == 'B', 'Cabin'] = 2

    data.loc[data.Cabin.str[0] == 'C', 'Cabin'] = 3

    data.loc[data.Cabin.str[0] == 'D', 'Cabin'] = 4

    data.loc[data.Cabin.str[0] == 'E', 'Cabin'] = 5

    data.loc[data.Cabin.str[0] == 'F', 'Cabin'] = 6

    data.loc[data.Cabin.str[0] == 'G', 'Cabin'] = 7

    data.loc[data.Cabin.str[0] == 'T', 'Cabin'] = 8



cabin(train_full)

cabin(test_full)
train_full.info()
train_full.head()
def name_preprocess(df):

    status_series = df.Name.str.split(',').str[1].str.split('.').str[0].str.strip().rename('Status')

    status_series.replace(to_replace=

                ['Col','Major','Mlle','Mme','Sir','Don','Dona','Capt','Jonkheer','Lady','Ms','the Countess'],

                value='Other', inplace=True)

    return status_series



train_status = name_preprocess(train_full)

test_status = name_preprocess(test_full)

m = {'child' : 0, 'adult' : 1}

train_full["AgeCat"]= pd.cut(train_full["Age"], bins=[0,15,max(train_full["Age"]+1)],

                               labels=['child','adult']).map(m).fillna(1).astype(int)

test_full["AgeCat"]= pd.cut(test_full["Age"], bins=[0,15,max(test_full["Age"]+1)],

                              labels=['child','adult']).map(m).fillna(1).astype(int)

# Drop Cabin - lots of nan data, hard to encode.

# Drop Ticket - high cardinality data, low correlation with survival

# Drop PassengerId - metafield

X = train_full.drop(columns=['PassengerId','Ticket','Name', 'Age']).join(train_status)

X_test = test_full.drop(columns=['PassengerId','Ticket','Name', 'Age']).join(test_status)
X.head()
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score, train_test_split



X_train, X_valid, y_train, y_valid = train_test_split(X,y)
numerical =  ['Fare']

categories = ['Embarked', 'Status']



num_pipe = Pipeline(steps=[

    ('mean', SimpleImputer(strategy='mean'))

])



cat_pipe = Pipeline(steps=[

        ('freq', SimpleImputer(strategy='most_frequent')),

        ('one hot', OneHotEncoder(handle_unknown='ignore', sparse=False))

])



preprocess = ColumnTransformer(remainder='passthrough', transformers=[

        ('num', num_pipe, numerical),

        ('cat', cat_pipe, categories)

])



from xgboost import XGBClassifier





# Static validation set



preprocess.fit(X_train)



xgb_model = XGBClassifier(n_estimators=200, learning_rate=0.05)

xgb_model.fit(preprocess.transform(X_train), y_train, early_stopping_rounds=5,

              eval_set=[(preprocess.transform(X_valid), y_valid)], verbose=False)



xgb_pred_1 = xgb_model.predict(preprocess.transform(X_valid))

print("Accuracy with static validation set using XGB: {}".format(accuracy_score(xgb_pred_1, y_valid)))



xgb_1 = xgb_model.predict(preprocess.transform(X_test))

xgb_1 = np.greater(xgb_1,0.5).astype(int)

df = pd.DataFrame({'PassengerId': test_full['PassengerId'], 'Survived': xgb_1})

df.to_csv('xgb_static_val_predictions.csv', index=False)





# Validation rotation using 5 folds



preprocess.fit(X)

xgb_model_2 = XGBClassifier(n_estimators=100, learning_rate=0.05)

scores = cross_val_score(xgb_model_2, preprocess.transform(X), y, cv=5, scoring='accuracy')

print('Mean Accuracy with rotating validation across 5 folds: {}'.format(scores.mean()))



xgb_2 = xgb_model.predict(preprocess.transform(X_test))

xgb_2 = np.greater(xgb_1,0.5).astype(int)

df = pd.DataFrame({'PassengerId': test_full['PassengerId'], 'Survived': xgb_2})

df.to_csv('xgb_rotating_val_predictions.csv', index=False)
from lightgbm import LGBMClassifier



lgbm_model = LGBMClassifier(max_depth=-1,

                            n_estimators=3000,

                            learning_rate=0.1,

                            colsample_bytree=0.2,

                            objective='binary', 

                            n_jobs=-1)



preprocess.fit(X_train)



lgbm_model.fit(preprocess.transform(X_train), y_train, eval_metric='auc', eval_set=[(preprocess.transform(X_valid), y_valid)], 

          verbose=100, early_stopping_rounds=100)



lgbm_pred = xgb_model.predict(preprocess.transform(X_valid))

print("Accuracy with using LightGBM: {}".format(accuracy_score(lgbm_pred, y_valid)))



lgbm_pred = xgb_model.predict(preprocess.transform(X_test))

lgbm_pred = np.greater(xgb_1,0.5).astype(int)



dflgbm = pd.DataFrame({'PassengerId': test_full['PassengerId'], 'Survived': lgbm_pred})

dflgbm.to_csv('lgbm_predictions.csv', index=False)