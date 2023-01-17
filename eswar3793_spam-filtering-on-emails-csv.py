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
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.neighbors import NearestNeighbors

from sklearn import tree
data=pd.read_csv('/kaggle/input/emails.csv')
data['text'].replace({'Subject:': ''}, inplace=True, regex=True)

X=data['text']

y=data['spam']

vectorizer=CountVectorizer()

X = vectorizer.fit_transform(X)

print(X.toarray())
X_train,X_test,Y_train,Y_test=train_test_split(X, y, test_size = 0.2, random_state = 0)
lr=LogisticRegression()

nb=GaussianNB()

svm=SVC(kernel='rbf')

nn=NearestNeighbors(n_neighbors=2, algorithm='ball_tree')

dt=tree.DecisionTreeClassifier()
lr.fit(X_train,Y_train)

# nb.fit(X_train,Y_train)

svm.fit(X_train,Y_train)

nn.fit(X_train,Y_train)

dt.fit(X_train,Y_train)
lr_predicted=lr.predict(X_test)

svm_predicted=svm.predict(X_test)

nn_predicted=nn.kneighbors(X_test)

dt_predicted=dt.predict(X_test)
lr_cls_report=classification_report(Y_test,lr_predicted)

svm_cls_report=classification_report(Y_test,svm_predicted)

# nb_cls_report=classification_report(Y_test,nb_predicted)

dt_cls_report=classification_report(Y_test,dt_predicted)

print("Logistic regression : "+lr_cls_report)

print("SVM : "+svm_cls_report)

# print("Naive Bayes : "+nb_cls_report)

print("Decision tree : "+dt_cls_report)