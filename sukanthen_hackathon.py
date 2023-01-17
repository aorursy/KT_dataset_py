import numpy as np

import pandas as pd

import glob

import os

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
train = pd.read_csv('../input/avhranalytics/train_jqd04QH.csv')

test = pd.read_csv('../input/avhranalytics/test_KaymcHn.csv')

train.head()
data = pd.read_csv('../input/hr-analytics/train_hr.csv')

data.head()
data.info()
data = data.drop(['enrollee_id','city','company_size'],axis=1)

data.head()
X = data.drop('target',axis=1)

y = data['target']

X.shape, y.shape
from sklearn.ensemble import RandomForestClassifier

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")

imputer.fit(X)

X = imputer.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.10)
exported_pipeline = RandomForestClassifier(bootstrap=True, criterion="entropy", max_features=0.35000000000000003, min_samples_leaf=12, min_samples_split=12, n_estimators=100)



exported_pipeline.fit(X_train,y_train)
results
results.min()
accuracy_score(y_test, results)
from sklearn.metrics import roc_auc_score

roc_auc_score(y_test, results)
exported_pipeline.fit(X,y)
test = pd.read_csv('../input/test-data/test_hr.csv')

test.head()
data = test.drop(['enrollee_id','city','company_size'],axis=1)

data.head()

print(data.shape)
imputer = SimpleImputer(strategy="median")

imputer.fit(data)

test_data = imputer.transform(data)
pred = exported_pipeline.predict(test_data)
my_sub = pd.DataFrame({'enrollee_id':test.enrollee_id, 'target': pred})

my_sub.to_csv('Sukans Submission.csv', index=False)