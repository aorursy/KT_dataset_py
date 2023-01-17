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
import numpy as np

import pandas as pd



from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import RidgeClassifier



from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score



from sklearn.model_selection import GridSearchCV





from sklearn.metrics import accuracy_score, f1_score, roc_auc_score



import sys

if not sys.warnoptions:

    import warnings

    warnings.simplefilter("ignore")
df=pd.read_csv('../input/train.csv', header=0, index_col=0)

df=df.drop(['Cabin', 'Ticket', 'Name'], axis=1).dropna()

df.head()
labels=df.Survived

Data=df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

Data=pd.get_dummies(Data)

dummies = pd.get_dummies(Data['Pclass'], prefix='pc')

Data = pd.concat([Data, dummies],axis=1).drop(['Pclass'],axis=1)
X_train,  X_test, y_train, y_test = train_test_split(Data, labels, test_size=0.25, random_state=1)



LR=LogisticRegression()

DT=DecisionTreeClassifier()

KN=KNeighborsClassifier()

NB=GaussianNB()

ridge = RidgeClassifier()



ridge_par={'alpha':[0.001, 0.01, 0.1, 1, 2, 10, 20, 30], 'fit_intercept':[True, False], 'normalize':[True, False]}

LR_par={'penalty':['l1', 'l2'], 'C':[1, 2.5, 2.8, 2.9, 3, 3.2]}

DT_par={ 'criterion':['gini', 'entropy'], 'min_samples_split': np.linspace(0.007, 0.08, 12)}

KN_par={'n_neighbors': [5,10,15,20,25,30,35,40,45,50,55,60], 'weights':['uniform', 'distance']}
lr_cv = GridSearchCV(LR, LR_par, cv=5, scoring='accuracy')

lr_cv.fit(X_train, y_train)



print(lr_cv.best_params_)

print(lr_cv.best_score_)





dt_cv = GridSearchCV(DT, DT_par, cv=5, scoring='accuracy')

dt_cv.fit(X_train, y_train)



print(dt_cv.best_params_)

print(dt_cv.best_score_)





kn_cv = GridSearchCV(KN, KN_par, cv=5, scoring='accuracy')

kn_cv.fit(X_train, y_train)



print(kn_cv.best_params_)

print(kn_cv.best_score_)



ridge_cv = GridSearchCV(ridge, ridge_par, cv=5, scoring='accuracy')

ridge_cv.fit(X_train, y_train)



print(ridge_cv.best_params_)

print(ridge_cv.best_score_)



cross_val_score(NB, X_train, y_train, cv=5, scoring='accuracy').mean()
LR=LogisticRegression(penalty='l1', C=2.5)

LR.fit(X_train, y_train)

print(LR.score(X_test, y_test))



DT=DecisionTreeClassifier(min_samples_split = 0.02)

DT.fit(X_train, y_train)

print(DT.score(X_test, y_test))



r=RidgeClassifier(alpha=0.001, fit_intercept=True, normalize = True)

r.fit(X_train, y_train)

print(r.score(X_test, y_test))
df=pd.read_csv('../input/test.csv', header=0, index_col=0)

df=df.drop(['Cabin', 'Ticket', 'Name'], axis=1)

Data=df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

Data=pd.get_dummies(Data)

dummies = pd.get_dummies(Data['Pclass'], prefix='pc')

Data = pd.concat([Data, dummies],axis=1).drop(['Pclass'],axis=1)

Data.head()
Data['Age']=Data['Age'].fillna(np.nanmean(Data['Age']))

Data['Fare']=Data['Fare'].fillna(np.nanmean(Data['Fare']))

holdout=pd.read_csv('../input/gender_submission.csv', header=0)



holdout_predictions=LR.predict(Data)

holdout_ids = holdout['PassengerId']

submission_df = {"PassengerId": holdout_ids,

                 "Survived": holdout_predictions}

submission = pd.DataFrame(submission_df)



submission.to_csv('titanic_submission.csv', index=False)