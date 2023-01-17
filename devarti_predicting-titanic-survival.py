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
train= pd.read_csv("../input/titanic/train.csv")
train.head()
test= pd.read_csv("/kaggle/input/titanic/test.csv")
train["Sex"]=train["Sex"].replace({"male":0,"female":1})
test["Sex"]=test["Sex"].replace({"male":0,"female":1})
train['Age']=train['Age'].fillna(train['Age'].mean())
test['Age']=test['Age'].fillna(test['Age'].mean())
train['FirstClass']=train['Pclass'].apply(lambda x: 1 if x == 1 else 0)
test['FirstClass']=test['Pclass'].apply(lambda x: 1 if x == 1 else 0)
train['SecondClass']=train['Pclass'].apply(lambda x: 1 if x == 2 else 0)
test['SecondClass']=test['Pclass'].apply(lambda x: 1 if x == 2 else 0)
features=train[['Sex', 'Age', 'FirstClass', 'SecondClass']]
testf=test[['Sex', 'Age', 'FirstClass', 'SecondClass']]
survival=train['Survived']
from sklearn.preprocessing import StandardScaler
s= StandardScaler()
trainx=s.fit_transform(features)
testx=s.transform(testf)
from sklearn import model_selection

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB
clf1 = LogisticRegression(random_state=1)

clf2 = RandomForestClassifier(random_state=1)

clf3 = GaussianNB()

clf4= SVC()



labels = ['Logistic Regression', 'Random Forest', 'Naive Bayes','SVM']



for clf, label in zip([clf1, clf2, clf3,clf4], labels):



    scores = model_selection.cross_val_score(clf, trainx, survival, cv=5, scoring='accuracy')

    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
from mlxtend.classifier import EnsembleVoteClassifier



eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3,clf4], weights=[1,1,1,1])



labels = ['Logistic Regression', 'Random Forest', 'Naive Bayes','SVM', 'Ensemble']



for clf, label in zip([clf1, clf2, clf3, clf4, eclf], labels):

    scores = model_selection.cross_val_score(clf, trainx, survival, cv=5, scoring='accuracy')

    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
eclf.fit(trainx,survival)
pred= eclf.predict(testx)
pred