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
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')

#print information of train and test datasets,learn datasets
print(train.info())
print(test.info())
# select predict index
selected_features=['Pclass','Sex','Age','Embarked','SibSp','Parch','Fare']
X_train=train[selected_features]
X_test=test[selected_features]
y_train=train['Survived']

# fill missing values
print(X_train['Embarked'].value_counts())
print(X_test['Embarked'].value_counts())
X_train['Embarked'].fillna('S',inplace=True)
X_test['Embarked'].fillna('S',inplace=True)
# fill 'Age','Fare'
X_train['Age'].fillna(X_train['Age'].mean(),inplace=True)
X_test['Age'].fillna(X_test['Age'].mean(),inplace=True)
X_test['Fare'].fillna(X_test['Fare'].mean(),inplace=True)

X_train.info()
X_test.info()
from sklearn.feature_extraction import DictVectorizer
dict_vec=DictVectorizer(sparse=False)
X_train=dict_vec.fit_transform(X_train.to_dict(orient='record'))
X_test =dict_vec.transform(X_test.to_dict(orient='record'))
# RandomForestClassifier model
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
# XGBClassifier model
from xgboost import XGBClassifier
xgbc=XGBClassifier()
# cross validation
from sklearn.cross_validation import cross_val_score
# cross validation score of RandomForestClassifier model
cross_val_score(rfc, X_train, y_train, cv=5).mean()
# cross validation score of XGBClassifier model
cross_val_score(xgbc, X_train, y_train, cv=5).mean()
rfc.fit(X_train,y_train)
rfc_y_predict=rfc.predict(X_test)
rfc_submission=pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': rfc_y_predict})
rfc_submission.to_csv('rfc_submission.csv', index=False)

xgbc.fit(X_train, y_train)
xgbc_y_predict=xgbc.predict(X_test)
xgbc_submission=pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': xgbc_y_predict})
xgbc_submission.to_csv('xgbc_submission.csv', index=False)
from sklearn.grid_search import GridSearchCV
# opti XGBClassifier model
max_depth=[2,3,4,5,6]
n_estimators=[100,300,500,700,900]
learning_rate=[0.05, 0.1, 0.25, 0.5, 1.0]
params_xgbc={'max_depth':max_depth, 'n_estimators':n_estimators, 'learning_rate':learning_rate}
xgbc_best=XGBClassifier()
gs_xgbc=GridSearchCV(xgbc_best, params_xgbc, n_jobs=-1, cv=5, verbose=1)
gs_xgbc.fit(X_train, y_train)

print(gs_xgbc.best_score_)
print(gs_xgbc.best_params_)

xgbc_best_y_predict=gs_xgbc.predict(X_test)
xgbc_best_submission=pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': xgbc_best_y_predict})
xgbc_best_submission.to_csv('xgbc_best_submission.csv',index=False)
rfc_best=RandomForestClassifier()
max_features=[2,3,4,5,6,7]
params_rfc={'max_depth':max_depth,'max_features':max_features}
gs_rfc=GridSearchCV(rfc_best, params_rfc, n_jobs=-1, cv=5, verbose=1)
gs_rfc.fit(X_train, y_train)

print(gs_rfc.best_score_)
print(gs_rfc.best_params_)

rfc_best_y_predict=gs_rfc.predict(X_test)
rfc_best_submission=pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': rfc_best_y_predict})
rfc_best_submission.to_csv('rfc_best_submission.csv',index=False)