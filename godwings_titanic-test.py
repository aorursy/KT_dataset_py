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
import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")

print(test.info())

print(train.info())
select_feature = ['Pclass','Sex','Age','Embarked','SibSp','Parch','Fare']

X_train = train[select_feature]

X_test = test[select_feature]



Y_train = train['Survived']



print(X_train['Embarked'].value_counts())

print(X_test['Embarked'].value_counts())
X_train['Embarked'].fillna('S',inplace = True)

X_test['Embarked'].fillna('S',inplace = True)



X_train['Age'].fillna(X_train['Age'].mean(), inplace = True)

X_test['Age'].fillna(X_test['Age'].mean(), inplace = True)

X_test['Fare'].fillna(X_test['Fare'].mean(), inplace = True)



X_train.info()

X_test.info()
from sklearn.feature_extraction import DictVectorizer

dict_vec = DictVectorizer(sparse = False)

X_train = dict_vec.fit_transform(X_train.to_dict(orient = 'record'))

dict_vec.feature_names_
X_test = dict_vec.transform(X_test.to_dict(orient = 'record'))

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

from xgboost import XGBClassifier

xgbc = XGBClassifier()

from sklearn.model_selection import cross_val_score

cross_val_score(rfc,X_train,Y_train,cv = 5).mean()
cross_val_score(xgbc,X_train,Y_train,cv = 5).mean()
rfc.fit(X_train,Y_train)

rfc_y_predict = rfc.predict(X_test)

rfc_submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':rfc_y_predict})



rfc_submission.to_csv('rfc_submission.csv',index = False)



xgbc.fit(X_train, Y_train)

xgbc_y_predict = xgbc.predict(X_test)

xgbc_submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':xgbc_y_predict})

xgbc_submission.to_csv('xgbc_submission.csv',index = False)
from sklearn.model_selection import GridSearchCV

params = {'max_depth':range(2,7),'n_estimators':range(100,1100,200),'learning_rate':[0.05,0.1,0.25,0.5,1.0]}

xgbc_best = XGBClassifier()

gs = GridSearchCV(xgbc_best, params, n_jobs = -1, cv = 5, verbose = 1)

gs.fit(X_train, Y_train)
print(gs.best_score_)

print(gs.best_params_)
xgbc_best_y_predict = gs.predict(X_test)

xgbc_best_submission = pd.DataFrame({'PassengerID':test['PassengerId']})

xgbc_best_submission.to_csv('xgbc_best_submission.csv', index = False)                         