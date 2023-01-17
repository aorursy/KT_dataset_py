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
loans = pd.read_csv('/kaggle/input/predicting-who-pays-back-loans/loan_data.csv')
loans.head()
loans.info()
loans.isnull().sum()

# NO MISSING VALUES

# with no missing values, we'll turn to the columns and see if any should be dropped or edited.
# not fully paid, with a score of 1 or 0, will be our target value and the variable we want to predict
# let's observe the one 'object' data type

loans['purpose'].nunique()

# With 7 unique values that may correlate with our target value, we'll turn it into a dummy variable
purpose_ = pd.get_dummies(loans['purpose'],drop_first=True)

public_record  = pd.get_dummies(loans['pub.rec'],drop_first=True)
# drop the original columns that we're replacing with dummy variables 

loans.drop(['purpose','pub.rec'],axis=1,inplace=True)
# dummy decider

# loans['pub.rec'].unique()
loans = pd.concat([loans,purpose_,public_record],axis=1)

loans.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(loans.drop('not.fully.paid',axis=1), 

                                                    loans['not.fully.paid'], test_size=0.30)
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
logistic_confusion_matrix = confusion_matrix(y_test,predictions)

logistic_classification_report = classification_report(y_test,predictions)



print(confusion_matrix(y_test,predictions))

print(classification_report(y_test,predictions))
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()

dtree.fit(X_train,y_train)
predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
decision_tree_confusion_matrix = confusion_matrix(y_test,predictions)

decision_tree_classification_report = (classification_report(y_test,predictions))



print(confusion_matrix(y_test,predictions))

print(classification_report(y_test,predictions))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=600)
rfc.fit(X_train,y_train)
predictions = rfc.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
random_forests_confusion_matrix = confusion_matrix(y_test,predictions)

random_forests_classification_report = classification_report(y_test,predictions)



print(confusion_matrix(y_test,predictions))

print(classification_report(y_test,predictions))
print(logistic_confusion_matrix) 

print(logistic_classification_report)



print(decision_tree_confusion_matrix) 

print(decision_tree_classification_report)



print(random_forests_confusion_matrix) 

print(random_forests_classification_report)