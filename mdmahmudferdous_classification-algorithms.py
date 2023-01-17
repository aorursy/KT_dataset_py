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

import warnings

warnings.filterwarnings('ignore')
df=pd.read_csv('/kaggle/input/titanic/titanic_cleaned.csv')

df.head()
df.shape
df.info()
df.isnull().sum()
X=df.drop(columns=['Survived'])

y=df[['Survived']]

print(X.shape)

print(y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)

print(y_test.shape)
from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB



lr=LogisticRegression().fit(X_train, y_train)

sgd=SGDClassifier().fit(X_train, y_train)

rf=RandomForestClassifier().fit(X_train, y_train)

dtc=DecisionTreeClassifier().fit(X_train, y_train)

svc=SVC().fit(X_train, y_train)

nb=GaussianNB().fit(X_train, y_train)



y_lr=lr.predict(X_test)

y_sgd=sgd.predict(X_test)

y_rf=rf.predict(X_test)

y_dtc=dtc.predict(X_test)

y_svc=svc.predict(X_test)

y_nb=nb.predict(X_test)
from sklearn.metrics import classification_report

print('Logistic Regression: ',classification_report(y_lr, y_test) )

print('Stochastic Gradient Descent: ',classification_report(y_sgd, y_test) )

print('Random Forest: ',classification_report(y_rf, y_test) )

print('Decision Tree: ',classification_report(y_dtc, y_test) )

print('Support Vector: ',classification_report(y_svc, y_test) )

print('Naive Bayes: ',classification_report(y_nb, y_test) )