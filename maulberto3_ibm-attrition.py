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



from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

%config InlineBackend.figure_format ='retina'
df = pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')

print(df.shape)

df.head()
# Y true, enconding it to binary

y = pd.DataFrame(np.zeros(shape=(df.shape[0], 1)))

y[ df['Attrition']=='No'] = 0

y[ df['Attrition']=='Yes'] = 1

print(y.shape)

y.head()
df2 = df.copy()

df2 = df2.drop('Attrition', axis=1) # df only X variables, y is target

df2.shape
df.describe() # shows only numerical variables statistics
cols = df.describe().columns

print(type(cols))

cols # column names of numerical variables
x1 = df[cols].copy() # training sub dataset 1, only numerical variables, copy() to prevent memory miss-assignment

print(x1.shape)

x1.head()
x2 = df.copy() # training sub dataset 2, only dummied-categorical variables, copy() to prevent memory miss-assignment

x2 = x2.drop(x1.columns, axis=1)

print(x2.columns, x2.shape)

x2 = pd.get_dummies(x2) # dummify-ing categ vars

print(x2.shape)

x2.head()
# joining x1 and x2 training datasets

x = x1.join(x2)

print(x.shape)

x.head()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import metrics
# Base model

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=123)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
yte_pred = logreg.predict(X_test)

print(yte_pred.shape)

print('yte_pred.mean():', yte_pred.mean())

print('y_test.values.mean():', y_test.values.mean())
print(metrics.classification_report(y_test, yte_pred))
print(metrics.confusion_matrix(y_test, yte_pred))

print('Accuracy:', metrics.accuracy_score(y_test, yte_pred))

#print('Accuracy:', (952+89)/X_train.shape[0])

print('Precision:', metrics.precision_score(y_test, yte_pred))

#print('Precision:', (952)/(952+110)) # this is the one that I ned to fix...

print('Recall:', metrics.recall_score(y_test, yte_pred))

#print('Recall:', 89/(89+110))

print('F1:', metrics.f1_score(y_test, yte_pred))

#print('F1:', 2 * metrics.precision_score(y_train, y_pred) * metrics.recall_score(y_train, y_pred) / (metrics.precision_score(y_train, y_pred) + metrics.recall_score(y_train, y_pred)))
# TODO: interpret metrics



# Accuracy is all correct predictions over all predictions



# Precision is 



# Recall is 44 / 48 = 
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_train, y_train)

yte_pred = nb.predict(X_test)

print(metrics.confusion_matrix(y_test, yte_pred))

print('Accuracy:', metrics.accuracy_score(y_test, yte_pred))

print('Precision:', metrics.precision_score(y_test, yte_pred))

print('Recall:', metrics.recall_score(y_test, yte_pred))

print('F1:', metrics.f1_score(y_test, yte_pred))
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(X_train, y_train)

yte_pred = rf.predict(X_test)

print(metrics.confusion_matrix(y_test, yte_pred))

print('Accuracy:', metrics.accuracy_score(y_test, yte_pred))

print('Precision:', metrics.precision_score(y_test, yte_pred))

print('Recall:', metrics.recall_score(y_test, yte_pred))

print('F1:', metrics.f1_score(y_test, yte_pred))
# perfect test fit with ensemble method!
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dt.fit(X_train, y_train)

yte_pred = dt.predict(X_test)

print(metrics.confusion_matrix(y_test, yte_pred))

print('Accuracy:', metrics.accuracy_score(y_test, yte_pred))

print('Precision:', metrics.precision_score(y_test, yte_pred))

print('Recall:', metrics.recall_score(y_test, yte_pred))

print('F1:', metrics.f1_score(y_test, yte_pred))
# also with simple desicion tree
from sklearn.linear_model import RidgeClassifier

rc = RidgeClassifier(alpha=100)

rc.fit(X_train, y_train)

yte_pred = rc.predict(X_test)

print(metrics.confusion_matrix(y_test, yte_pred))

print('Accuracy:', metrics.accuracy_score(y_test, yte_pred))

print('Precision:', metrics.precision_score(y_test, yte_pred))

print('Recall:', metrics.recall_score(y_test, yte_pred))

print('F1:', metrics.f1_score(y_test, yte_pred))

print(rc.coef_)