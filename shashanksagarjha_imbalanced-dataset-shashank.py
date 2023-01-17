# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import seaborn as sns
%matplotlib inline

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sample_submission = pd.read_csv('../input/sample_submission.csv')
train.head()
sns.countplot(train['target'])
#Highly Imbalanced,Will need under_sampling and over_sampling
train.head()
train=train.drop(columns='id',axis=1)
test=test.drop(columns='id',axis=1)
#Checking for nulls
print('Train data null details \n')
print(train.isnull().sum().sort_values(ascending=False).head(5))
print('\n Test data null details')
print(test.isnull().sum().sort_values(ascending=False).head(5))
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

X = train.drop(columns='target',axis=1)
y = train['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Model fitting,training and checking accuracy
model = XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
# Class count
count_class_0, count_class_1 = train.target.value_counts()

# Divide by class
class_0 = train[train['target'] == 0]
class_1 = train[train['target'] == 1]
class_0_under = class_0.sample(count_class_1)
train_under = pd.concat([class_0_under, class_1], axis=0)

print('Random under-sampling:')
print(train_under.target.value_counts())

sns.countplot(train_under['target'])
X_under = train_under.drop(columns='target',axis=1)
y_under = train_under['target']
X_train_under, X_test_under, y_train_under, y_test_under = train_test_split(X_under, y_under, test_size=0.2, random_state=1)
# Model fitting,training and checking accuracy
model = XGBClassifier()
model.fit(X_train_under, y_train_under)
y_pred_under = model.predict(X_test_under)

accuracy = accuracy_score(y_test_under, y_pred_under)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

print(classification_report(y_test_under, y_pred_under))
print(confusion_matrix(y_test_under, y_pred_under))
# atleast there is improvement in class 1's performance
class_1_over = class_1.sample(count_class_0, replace=True)
train_over = pd.concat([class_0, class_1_over], axis=0)

print('Random over-sampling:')
print(train_over.target.value_counts())

sns.countplot(train_over['target'])
X_over = train_over.drop(columns='target',axis=1)
y_over = train_over['target']
X_train_over, X_test_over, y_train_over, y_test_over = train_test_split(X_over, y_over, test_size=0.2, random_state=1)

# Model fitting,training and checking accuracy
model = XGBClassifier(max_depth=5)
model.fit(X_train_over, y_train_over)
y_pred_over = model.predict(X_test_over)

accuracy = accuracy_score(y_test_over, y_pred_over)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

print(classification_report(y_test_over, y_pred_over))
print(confusion_matrix(y_test_over, y_pred_over))
