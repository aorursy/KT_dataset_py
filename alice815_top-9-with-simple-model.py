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
import seaborn as sns

import matplotlib.pyplot as plt
gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test_raw = pd.read_csv("../input/titanic/test.csv", index_col='PassengerId')

train_raw = pd.read_csv("../input/titanic/train.csv",index_col='PassengerId')
X=train_raw.drop('Survived', axis=1)

y=train_raw['Survived']

X_cb=pd.concat([X, test_raw], axis=0)

test=test_raw

train_idx=list(train_raw.index)

test_idx=list(test_raw.index)
X_cb.info()
X_cb.isnull().sum().sort_values(ascending=False)
test.groupby(['Pclass','Embarked'])['Fare'].describe()
class_fare=test.groupby(['Pclass','Embarked'])['Fare'].mean() 

test['Fare'].fillna(class_fare[3]['S'], inplace=True) #The passenger with missing fare was embarked from 'S', and Pclass is 3
X['Embarked'].value_counts()
X['Embarked'].fillna(X['Embarked'].mode()[0], inplace=True)
g=sns.FacetGrid(train_raw,col='Survived', height=5)

g.map(plt.hist, 'Age', bins=20)
X[['Last','First']]=X['Name'].str.split(',', expand=True)

X[['Title', 'FName']]=X['First'].str.split('.', n=1, expand=True)

X['Title']=X['Title'].str.strip()

test[['Last','First']]=test['Name'].str.split(',', expand=True)

test[['Title', 'FName']]=test['First'].str.split('.', n=1, expand=True)

test['Title']=test['Title'].str.strip()
title=X.groupby('Title')['Age'].describe()

title
X['Adult']=''

for i in list(X.index):

    if X['Age'].loc[i]>8:

        X['Adult'].loc[i]=1

    elif X['Age'].loc[i]<=8:

        X['Adult'].loc[i]=0

    else:

        if X['Title'].loc[i]=='Master':

            X['Adult'].loc[i]=0

        else: 

            X['Adult'].loc[i]=1   

test['Adult']=''

for i in list(test.index):

    if test['Age'].loc[i]>8:

        test['Adult'].loc[i]=1

    elif test['Age'].loc[i]<=8:

        test['Adult'].loc[i]=0

    else:

        if test['Title'].loc[i]=='Master':

            test['Adult'].loc[i]=0

        else: 

            test['Adult'].loc[i]=1
X['Family_Size']=X['SibSp']+X['Parch']+1

X["With_Family"]=X.apply(lambda row: 1 if row['Family_Size']>1 else 0, axis=1)

test['Family_Size']=test['SibSp']+test['Parch']+1

test["With_Family"]=test.apply(lambda row: 1 if row['Family_Size']>1 else 0, axis=1)

count=X['Ticket'].value_counts().to_frame().reset_index()

count.columns=['Ticket','Person_on_Ticket']

X=pd.merge(X,count,how='left',on='Ticket').set_index(X.index)

count_t=test['Ticket'].value_counts().to_frame().reset_index()

count_t.columns=['Ticket','Person_on_Ticket']

test=pd.merge(test,count_t,how='left',on='Ticket').set_index(test.index)
X.drop(['Name','Age','Ticket','First','FName','Cabin'], axis=1, inplace=True)

test.drop(['Name','Age','Ticket','First','FName','Cabin'], axis=1, inplace=True)
X=pd.get_dummies(X)

test=pd.get_dummies(test)

X.head()
only_in_train_col=list(set(X.columns)-set(test.columns))

only_in_test_col=list(set(test.columns)-set(X.columns))

for col in only_in_train_col:

    test[col]=0

for col in only_in_test_col:

    test.drop(col, axis=1, inplace=True)

test=test[X.columns]
print(X.shape, test.shape)
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn import svm

#from sklearn.linear_model import SGDClassifier

#from sklearn.neighbors import KNeighborsClassifier

#from sklearn.gaussian_process import GaussianProcessClassifier

#from sklearn.naive_bayes import GaussianNB

#from catboost import CatBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier



kf=KFold(n_splits=5, random_state=0, shuffle=True)

X_train,X_valid, y_train,y_valid=train_test_split(X,y, test_size=0.3, random_state=0)

def cv_scores(model, X):

    cv_scores=cross_val_score(model, X, y, cv=5)

    return cv_scores



import warnings

warnings.filterwarnings(action="ignore")
dt=DecisionTreeClassifier(random_state=0)

rf=RandomForestClassifier(n_estimators=100, max_features=None, max_depth=None)

ada=AdaBoostClassifier(n_estimators=100,learning_rate=0.4)

gdb=GradientBoostingClassifier(n_estimators=100, learning_rate=0.4)

xgb=XGBClassifier(learning_rate=0.5)
cv_score_dt=cv_scores(dt,X).mean()

cv_score_rf=cv_scores(rf,X).mean()

cv_score_ada=cv_scores(ada,X).mean()

cv_score_gdb=cv_scores(gdb,X).mean()

cv_score_xgb=cv_scores(xgb,X).mean()
print(' dt cv score: ',cv_scores(dt,X).mean(), 'std: ',cv_scores(dt,X).std(),'\n',

    'rf cv score: ',cv_scores(rf,X).mean(), 'std: ',cv_scores(rf,X).std(),'\n',

     'ada cv score: ',cv_scores(ada,X).mean(), 'std: ',cv_scores(ada,X).std(),'\n',

     'gdb cv score: ',cv_scores(gdb,X).mean(), 'std: ',cv_scores(gdb,X).std(),'\n',

     'xgb cv score: ',cv_scores(xgb,X).mean(), 'std: ',cv_scores(xgb,X).std(),

     )
dt.fit(X_train,y_train)

rf.fit(X_train,y_train)

ada.fit(X_train,y_train)

gdb.fit(X_train,y_train)

xgb.fit(X_train,y_train)
y_dt=dt.predict(X_valid)

y_rf=rf.predict(X_valid)

y_ada=ada.predict(X_valid)

y_gdb=gdb.predict(X_valid)

y_xgb=xgb.predict(X_valid)
print(' dt accuracy: ',accuracy_score(y_valid,y_dt), '\n',

    'rf accuracy: ',accuracy_score(y_valid,y_rf), '\n',

     'ada accuracy: ',accuracy_score(y_valid,y_ada), '\n',

     'gdb accuracy: ',accuracy_score(y_valid,y_gdb), '\n',

     'xgb accuracy: ',accuracy_score(y_valid,y_xgb), 

     )
test_pred=gdb.predict(test)
output=pd.DataFrame({'PassengerId': test.index, 'Survived': test_pred})

output.to_csv('my_submission.csv', index=False)