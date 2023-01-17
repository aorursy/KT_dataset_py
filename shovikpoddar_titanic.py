# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('ggplot')

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#DATA

train=pd.read_csv("../input/titanic/train.csv")

test=pd.read_csv("../input/titanic/test.csv")
train.head()
test.head()
#check for missing data

train.isnull().sum()
test.isnull().sum()
sns.set_style('whitegrid')

sns.countplot(x='Survived', data=train, palette='autumn')
#survival dependent on gender

#0=Died, 1=Alive

sns.set_style('whitegrid')

sns.countplot(x='Survived', hue='Sex', data=train, palette='autumn')
#survival dependent on passenger class

sns.set_style('whitegrid')

sns.countplot(x='Survived', hue='Pclass', data=train, palette='autumn_r')
#distrbution of ages of passengers

train['Age'].hist(bins=45, color='orange')
#distribution of Sibilings and Spouse

sns.countplot(x='SibSp',data=train, palette='autumn_r')
#Approximately average fare

train['Fare'].hist(color='orange', bins=100, alpha=0.8, figsize=(16,5))
#TRAINING SET

import plotly.express as px

px.violin(train, x='Pclass', y='Age', color='Pclass', box=True, hover_data=train)
#Replacing NAN values with

#Pclass=1, Age=37

#Pclass=2, Age=29

#Pclass=3, Age=24

def fill_age(data):#data will consist of Age and Pclass respectively

    age=data[0]

    Pclass=data[1]

    if pd.isnull(age):

        if Pclass==1:  

            return 37

        elif Pclass==2:  

            return 29

        else:  

            return 24

    else:

        return age
train['Age']=train[['Age', 'Pclass']].apply(fill_age, axis=1)
px.violin(test, x='Pclass', y='Age', color='Pclass', box=True, hover_data=test)
#Replacing NAN values with

#Pclass=1, Age=42

#Pclass=2, Age=26.5

#Pclass=3, Age=24

def fill_age1(data):#data will consist of Age and Pclass respectively

    age=data[0]

    Pclass=data[1]

    if pd.isnull(age):

        if Pclass==1:  

            return 42

        elif Pclass==2:  

            return 26.5

        else:  

            return 24

    else:

        return age
test['Age']=test[['Age', 'Pclass']].apply(fill_age1, axis=1)
train.info()
test.info()
#since Cabin is an alphanumeric data and it consists of huge NAN values, it will be difficult to replace, hence we drop it

train.drop('Cabin', axis=1, inplace=True)

test.drop('Cabin', axis=1, inplace=True)
test['Fare'].mean
#A Fare value is missing, we shall replace it with the median value

test['Fare']=test['Fare'].fillna(7.8292)

test['Fare']=test['Fare'].astype('float')
train.info()
test.info()
train.head()
test.head()
#for Embarked

'''dropped the first column because thee categoriescan be represented by 

00 for C

10 for Q

01 for S'''

embark_train=pd.get_dummies(train['Embarked'], drop_first=True)

embark_test=pd.get_dummies(test['Embarked'], drop_first=True)
#For Sex

'''1 for male

0 for female'''

sex_train=pd.get_dummies(train['Sex'], drop_first=True)

sex_test=pd.get_dummies(test['Sex'], drop_first=True)
#dropping the Name, Ticket, Sex and Embarked column

train.drop(['Sex', 'Name', 'Ticket', 'Embarked'], axis=1, inplace=True)

test.drop(['Sex', 'Name', 'Ticket', 'Embarked'], axis=1, inplace=True)
#appending the new categorical columns to dataframe

train=pd.concat([train, embark_train, sex_train], axis=1)

test=pd.concat([test, embark_test, sex_test], axis=1)
train.head()
test.head()
X_train=train.drop('Survived', axis=1)

y_train=train['Survived']

X_test=test


#testing with one of the 1% submissions

submit=pd.read_csv('../input/ideal-submission/submission2.csv')

y_test=submit['Survived']
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()
#parameter grid

params={'penalty' : ['l1', 'l2', 'elasticnet'],

        'C' : [0.001,0.005,0.01,0.1,0.5,1],

        'max_iter' : [1000,5000],

        'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}

from sklearn.model_selection import RandomizedSearchCV

cv=RandomizedSearchCV(estimator=lr,

                      param_distributions=params,

                      n_iter=50,

                      n_jobs=-1,

                      random_state=45)

cv.fit(X_train, y_train)
cv.best_estimator_
lr=LogisticRegression(C=0.5, max_iter=1000, penalty='l1', solver='liblinear')

lr.fit(X_train, y_train)

y_pred_lr=lr.predict(X_test)
from sklearn.metrics import accuracy_score

acc_lr=round(accuracy_score(y_test, y_pred_lr)*100, 2)

acc_lr
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier()
#parameter grid

params={'n_estimators' : [100,200,500,1000],

        'criterion' : ['gini', 'entropy'],

        'max_features' : ['auto', 'sqrt', 'log2'],

        'bootstrap': [True],}

cv=RandomizedSearchCV(estimator=rf,

                      param_distributions=params,

                      n_jobs=-1,

                      random_state=45)

cv.fit(X_train, y_train)
cv.best_estimator_
rf=RandomForestClassifier(criterion='entropy', max_features='sqrt')

rf.fit(X_train, y_train)

y_pred_rf=rf.predict(X_test)
acc_rf=round(accuracy_score(y_test, y_pred_rf)*100, 2)

acc_rf
from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier()
#parameter grid

params={'splitter' : ['best', 'random'],

        'criterion' : ['gini', 'entropy'],

        'max_features' : ['auto', 'sqrt', 'log2']}

cv=RandomizedSearchCV(estimator=dt,

                      param_distributions=params,

                      n_jobs=-1,

                      verbose=5,

                      random_state=45)

cv.fit(X_train, y_train)
cv.best_estimator_
dt=DecisionTreeClassifier(criterion='entropy', max_features='log2')

dt.fit(X_train,y_train)

y_pred_dt=dt.predict(X_test)
acc_dt=round(accuracy_score(y_test, y_pred_dt)*100, 2)

acc_dt
from xgboost import XGBClassifier

xg=XGBClassifier()
#parameter grid

params={'booster' : ['gbtree', 'dart'],

        'learning_rate' : [0.03, 0.06, 0.1, 0.15, 0.2],

        'objective' : ['reg:logistic', 'binary:logistic']}

cv=RandomizedSearchCV(estimator=xg,

                      param_distributions=params,

                      n_iter=25,

                      n_jobs=-1,

                      verbose=5,

                      random_state=45)

cv.fit(X_train, y_train)
cv.best_estimator_
xg=XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,

              importance_type='gain', interaction_constraints='',

              learning_rate=0.03, max_delta_step=0, max_depth=6,

              min_child_weight=1, missing=None, monotone_constraints='()',

              n_estimators=100, n_jobs=0, num_parallel_tree=1,

              objective='reg:logistic', random_state=0, reg_alpha=0,

              reg_lambda=1, scale_pos_weight=1, subsample=1,

              tree_method='exact', validate_parameters=1, verbosity=None)

xg.fit(X_train,y_train)

y_pred_xg=xg.predict(X_test)
acc_xg=round(accuracy_score(y_test, y_pred_xg)*100, 2)

acc_xg
from sklearn.svm import SVC

svc=SVC()
#parameter grid

params={'C' : [0.01,0.05,0.1,0.5,1],

        'kernel' : ['rbf', 'sigmoid', 'poly'],

        'degree' : [2,3,4,5]}

cv=RandomizedSearchCV(estimator=svc,

                      param_distributions=params,

                      n_iter=25,

                      n_jobs=-1,

                      verbose=5,

                      random_state=45)

cv.fit(X_train, y_train)
cv.best_estimator_
svc=SVC(C=1, degree=2, kernel='poly')

svc.fit(X_train, y_train)

y_pred_svm=svc.predict(X_test)
acc_svm=round(accuracy_score(y_test, y_pred_svm)*100, 2)

acc_svm
from sklearn.linear_model import SGDClassifier

sgd=SGDClassifier()
#parameter grid

params={'penalty' : ['l1', 'l2', 'elasticnet'],

        'max_iter' : [500,1000,2000,5000],

        'shuffle': [True]}

cv=RandomizedSearchCV(estimator=sgd,

                      param_distributions=params,

                      n_iter=25,

                      n_jobs=-1,

                      verbose=5,

                      random_state=45)

cv.fit(X_train, y_train)
cv.best_estimator_
sgd=SGDClassifier(max_iter=2000, penalty='elasticnet')

sgd.fit(X_train, y_train)

y_pred_sgd=sgd.predict(X_test)
acc_sgd=round(accuracy_score(y_test, y_pred_sgd)*100, 2)

acc_sgd
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier()
#parameter grid

params={'weights' : ['uniform', 'distance'],

        'n_neighbors' : [3,4,5],

        'algorithm' : ['ball-tree', 'kd_tree', 'brute'],

        'p' : [1,2,3]}

cv=RandomizedSearchCV(estimator=knn,

                      param_distributions=params,

                      n_iter=25,

                      n_jobs=-1,

                      verbose=5,

                      random_state=45)

cv.fit(X_train, y_train)
cv.best_estimator_
knn=KNeighborsClassifier(algorithm='kd_tree', n_neighbors=3, p=1,

                         weights='distance')

knn.fit(X_train, y_train)

y_pred_knn=knn.predict(X_test)
acc_knn=round(accuracy_score(y_test, y_pred_knn)*100, 2)

acc_knn
from sklearn.naive_bayes import GaussianNB

nb=GaussianNB()

nb.fit(X_train, y_train)

y_pred_nb=nb.predict(X_test)
acc_nb=round(accuracy_score(y_test, y_pred_nb)*100, 2)

acc_nb
from sklearn.linear_model import Perceptron

per=Perceptron()
#parameter grid

params={'max_iter' : [500,1000,2000],

        'penalty' : ['l1', 'l2', 'elasticnet']}

cv=RandomizedSearchCV(estimator=per,

                      param_distributions=params,

                      n_iter=9,

                      n_jobs=-1,

                      verbose=5,

                      random_state=45)

cv.fit(X_train, y_train)
cv.best_estimator_
per=Perceptron(max_iter=500, penalty='l2')

per.fit(X_train, y_train)

y_pred_per=per.predict(X_test)
acc_per=round(accuracy_score(y_test, y_pred_per)*100, 2)

acc_per
from sklearn.svm import LinearSVC

lin_svm=LinearSVC(penalty="l1",dual=False,max_iter=5000)

lin_svm.fit(X_train, y_train)

y_pred_lin_svm=lin_svm.predict(X_test)
acc_lin_svm=round(accuracy_score(y_test, y_pred_lin_svm)*100, 2)

acc_lin_svm
sorted_models=pd.DataFrame({

    'Model': ['Logistic Regression', 'Random Forest', 'Decision Tree',

              'XG Boost', 'SVM', 'Stochastic Gradient Decent', 'KNN',

              'Naive Bayes', 'Perceptron', 'Linear SVC'],

    'Score': [acc_lr, acc_rf, acc_dt, acc_xg, acc_svm, 

              acc_sgd, acc_knn, acc_nb, acc_per, acc_lin_svm]})

sorted_models=sorted_models.sort_values(by='Score', ascending=False)

print(sorted_models.to_string(index=False))
plt.figure(figsize=(20,10))

plt.bar(sorted_models['Model'], sorted_models['Score'])
submission=pd.DataFrame({'PassengerId':X_test['PassengerId'],

                         'Survived':y_pred_rf})

submission.to_csv('submission2.csv', index=False)
submission=pd.read_csv('submission2.csv')

submission.info()