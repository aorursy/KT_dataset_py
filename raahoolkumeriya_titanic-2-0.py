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
import numpy as np

import pandas as pd

import os

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import  LogisticRegressionCV, LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier ,XGBRFClassifier

from sklearn.svm import SVC
train_set = pd.read_csv("/kaggle/input/titanic/train.csv")

test_set = pd.read_csv("/kaggle/input/titanic/test.csv")

train_set.head()
train_set.isnull().values.any()
train_set.isnull().sum().sum()
train_set.describe()
train_set['Sex'].value_counts()
train_set['Embarked'].value_counts()
train_set.head()
train_set.drop(['PassengerId','Name','Cabin','Ticket','Embarked'],axis=1 ,inplace=True),

test_set.drop(['PassengerId','Name','Cabin','Ticket','Embarked'],axis=1, inplace=True)
train_set.head()
test_set.head()
train_set = pd.get_dummies(data= train_set , dummy_na = True,columns =['Sex'])

test_set = pd.get_dummies(data= test_set , dummy_na = True,columns =['Sex'])
train_set.head()
train_set.drop(['Sex_male','Sex_nan'],axis=1 ,inplace=True)

test_set.drop(['Sex_male','Sex_nan'],axis=1 ,inplace=True)
train_set.head()
train_set.isnull().sum()
train_set.fillna(train_set.mean(),inplace=True)
train_set.isnull().sum()
test_set.isnull().sum()
test_set.fillna(test_set.mean(),inplace=True)
test_set.isnull().sum()
X_train = train_set.drop(['Survived'],axis=1)

X_train.head()
y_train = train_set.pop('Survived')

y_train
X_test = test_set

X_test.head()
X_train.head()
sc_X = MinMaxScaler()

X_train_sc = sc_X.fit_transform(X_train)

X_validate = sc_X.transform(X_train_sc)

X_validate
sc_X = MinMaxScaler()

X_test_sc = sc_X.fit_transform(X_test)

X_test_validate = sc_X.transform(X_test_sc)

X_test_validate
models = [ LogisticRegressionCV(), SVC(), XGBClassifier() ,  XGBRFClassifier(), KNeighborsClassifier(), RandomForestClassifier()]

for clf in models:

    clf.fit(X_validate,y_train)

    y_pred = clf.predict(X_validate)

    cnf = confusion_matrix(y_train,y_pred)

    print(cnf)

    print(accuracy_score(y_train,y_pred)*100)

clf
from pprint import pprint

# Look at parameters used by our current forest

print('Parameters currently in use:\n')

pprint(clf.get_params())
from sklearn.model_selection import GridSearchCV

# Create the parameter grid based on the results of random search 



grid_parameter = {

    'n_estimators' : [ 100,500],

    'max_features' : ['auto', 'sqrt', 'log2'],

    'max_depth' : [10,20,30],

    'min_samples_split' : [2,5,10],

    'min_samples_leaf' : [1,2,3,4,5]

}


clf = RandomForestClassifier( random_state= 0 )

grid_search = GridSearchCV(estimator=clf, param_grid=grid_parameter, cv=5, verbose=2, n_jobs=-1)

#grid_search.fit(X_validate,y_train)

#grid_search.best_params_
clf = RandomForestClassifier(bootstrap=True, class_weight=None,

                                              criterion='gini', max_depth=None,

                                              max_features='auto',

                                              max_leaf_nodes=None,

                                              min_impurity_decrease=0.0,

                                              min_impurity_split=None,

                                              min_samples_leaf=50,

                                              min_samples_split=2,

                                              min_weight_fraction_leaf=0.0,

                                              n_estimators=100, n_jobs=None,

                                              oob_score=True, random_state=50,

                                              verbose=0, warm_start=False)

clf.fit(X_validate,y_train)

y_pred = clf.predict(X_validate)

cnf = confusion_matrix(y_train,y_pred)

print(cnf)

print(accuracy_score(y_train,y_pred)*100)
'''

clf = RandomForestClassifier()

clf.fit(X_validate,y_train)

y_pred = clf.predict(X_validate)

cnf = confusion_matrix(y_train,y_pred)

print(cnf)

print(accuracy_score(y_train,y_pred)*100)

'''
'''

clf  = LogisticRegression(solver='lbfgs')

clf.fit(X_validate,y_train)

y_pred = clf.predict(X_validate)

cnf = confusion_matrix(y_train,y_pred)

print(cnf)

print(accuracy_score(y_train,y_pred)*100)

'''
X_test_validate
clf
y_submission = clf.predict(X_test_validate)
y_submission
sub = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

sub['Survived']=y_submission

sub.head()
sub.to_csv('submission.csv',index=False)
!ls -lart #/kaggle/working/submissions.csv
final = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

final['Survived'].value_counts()
submitted = pd.read_csv('submission.csv')

submitted['Survived'].value_counts()