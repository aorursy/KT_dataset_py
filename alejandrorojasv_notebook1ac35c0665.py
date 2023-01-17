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
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv('/kaggle/input/titanic/train.csv')
X=data.drop('Survived',axis=1)
survived=data['Survived']
X.columns
X[['Sex','Embarked']]=X[['Sex','Embarked']].astype('category')
X['Sex']=X['Sex'].cat.codes
X['Embarked']=X['Embarked'].cat.codes
X=X.drop(['PassengerId','Name','Ticket','Cabin','Embarked'],axis=1)
#Getting only Age with values to get analyze the values the have

X_fullAge=X[X['Age'].isnull()==False]
X_fullAge.groupby(["Sex", "Pclass"])["Age"].median()
#Adding Age value depending on Sex and Class (median)

sex=[0,1]
Pclass=[1,2,3]

for s in sex:
    for c in Pclass:
        X.loc[ (X['Age'].isnull()==True) & (X['Sex']==s) & (X['Pclass']==c),'Age']=np.median(X_fullAge.loc[(X_fullAge['Sex']==s) & (X_fullAge['Pclass']==c),'Age'])
#Calculate something similar to the price that paid the hole family

X['Fam_Fare']=(X['Parch']+X['SibSp'])*X['Fare']
# entrenar el random forest
rf = RandomForestClassifier().fit(X, survived)

# se obtienen las importancias de las cols (Impurities)
importances = rf.feature_importances_
# se ordenan los indices
indices = np.argsort(importances)[::-1]
# se ordena el dataset
cols = X.columns[indices]

print(importances)
print(indices)
print(cols)
X.drop(['Parch'],axis=1,inplace=True)
rfc = RandomForestClassifier().fit(X, survived)
print('cross_val = ', cross_val_score(rfc, X, survived, cv=50).mean())

data_test=pd.read_csv('/kaggle/input/titanic/test.csv')
data_test.head()
data_test[['Sex','Embarked']]=data_test[['Sex','Embarked']].astype('category')
data_test['Sex']=data_test['Sex'].cat.codes


X_test=data_test[['Pclass', 'Sex', 'Age', 'SibSp', 'Fare','Parch']]

X_test_fullAge=X_test[X_test['Age'].isnull()==False]

sex=[0,1]
Pclass=[1,2,3]

for s in sex:
    for c in Pclass:
        X_test.loc[ (X_test['Age'].isnull()==True) & (X_test['Sex']==s) & (X_test['Pclass']==c),'Age']=np.median(X_test_fullAge.loc[(X_test_fullAge['Sex']==s) & (X_test_fullAge['Pclass']==c),'Age'])


X_test['Fare'].fillna(0,inplace=True)

X_test['Fam_Fare']=(X_test['Parch']+X_test['SibSp'])*X_test['Fare']

X_test.drop(['Parch'],axis=1,inplace=True)
data_test['Survived']=rfc.predict(X_test)

#data_test[['PassengerId','Survived']].to_csv('rfc_1.csv',index=False)
