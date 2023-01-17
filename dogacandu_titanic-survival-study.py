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
import pandas as pd
titanic_traindf=pd.read_csv('../input/train.csv')
titanic_testxdf=pd.read_csv('../input/test.csv')
titanic_testydf=pd.read_csv('../input/gender_submission.csv')

titanic_traindf.head(2)
titanic_traindf=titanic_traindf.drop('Name', axis=1)
titanic_traindf=titanic_traindf.drop('Ticket', axis=1)
titanic_traindf=titanic_traindf.drop('Cabin', axis=1)
titanic_traindf=titanic_traindf.drop('Embarked', axis=1)
titanic_testxdf=titanic_testxdf.drop('Name', axis=1)
titanic_testxdf=titanic_testxdf.drop('Ticket', axis=1)
titanic_testxdf=titanic_testxdf.drop('Cabin', axis=1)
titanic_testxdf=titanic_testxdf.drop('Embarked', axis=1)
titanic_traindf.head(2)
titanic_groupby=titanic_traindf.groupby('Sex')
titanic_groupby.Survived.sum()/titanic_groupby.Survived.count()
titanic_traindf['Sex'].replace(to_replace=['female','male'], value=[0,1],inplace=True)
titanic_testxdf['Sex'].replace(to_replace=['female','male'], value=[0,1],inplace=True)
titanic_traindf.Age.isna().sum()
X=titanic_traindf[['Pclass','Sex','Age', 'SibSp','Parch','Fare']]
y=titanic_traindf['Survived']
test_X=titanic_testxdf[['Pclass','Sex','Age','SibSp', 'Parch','Fare']]
from xgboost import XGBClassifier
model=XGBClassifier(n_estimators=1000,learning_rate=0.05)
model.fit(X,y, early_stopping_rounds=5,eval_set=[(X, y)],verbose=False)
predicted = model.predict(test_X)

from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
imputer = Imputer()
X_imputed=pd.DataFrame(imputer.fit_transform(X))
test_X_imputed=pd.DataFrame(imputer.fit_transform(test_X))
X_imputed.columns=['Pclass','Sex','Age', 'SibSp','Parch','Fare']
test_X_imputed.columns=['Pclass','Sex','Age', 'SibSp','Parch','Fare']
rf = RandomForestClassifier()
rf.fit(X_imputed, y)
predicted = model.predict(test_X_imputed)
from sklearn import neighbors
knn = neighbors.KNeighborsClassifier(n_neighbors=1)
knn.fit(X_imputed, y)
predicted = knn.predict(test_X_imputed)