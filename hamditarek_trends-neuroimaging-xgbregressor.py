import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from xgboost.sklearn import XGBRegressor

import xgboost as xgb

import os
loading = pd.read_csv('/kaggle/input/trends-assessment-prediction/loading.csv')

features = pd.read_csv('/kaggle/input/trends-assessment-prediction/train_scores.csv')

submission = pd.read_csv('/kaggle/input/trends-assessment-prediction/sample_submission.csv')

fnc = pd.read_csv("/kaggle/input/trends-assessment-prediction/fnc.csv")

score = pd.read_csv("/kaggle/input/trends-assessment-prediction/train_scores.csv")
loading = pd.merge(fnc,loading, on="Id", how="inner") 
features.mean()

features = features.fillna(features.mean())
training_labels = ['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']
X_train = loading[loading.columns[1:]].head(n=5877)

y_train = features['age']

X_test = loading[loading.columns[1:]].tail(n=5877)
ls = submission['Predicted'].tolist()
clf = xgb.XGBRegressor(

    n_estimators=500,

    learning_rate=0.05,

    subsample=0.9,

    colsample_bytree=0.9,

    missing=-999,

    random_state=2019  

)
%time clf.fit(X_train, y_train)

preds = clf.predict(X_test)



j=0

for i in range(0, len(ls), 5):

    ls[i] = preds[j]

    j=j+1
print(len(ls))

print(len(preds))
y_train = features['domain1_var1'].values

%time clf.fit(X_train, y_train)

preds = clf.predict(X_test)

j=0

for i in range(1, len(ls), 5):

    ls[i] = preds[j]

    j=j+1
y_train = features['domain1_var2']

%time clf.fit(X_train, y_train)

preds = clf.predict(X_test)

j=0

for i in range(2, len(ls), 5):

    ls[i] = preds[j]

    j=j+1
y_train = features['domain2_var1']

%time clf.fit(X_train, y_train)

preds = clf.predict(X_test)

j=0

for i in range(3, len(ls), 5):

    ls[i] = preds[j]

    j=j+1
y_train = features['domain2_var2'].values

%time clf.fit(X_train, y_train)

preds = clf.predict(X_test)

j=0

for i in range(4, len(ls), 5):

    ls[i] = preds[j]

    j=j+1
submission['Predicted'] = ls
submission.to_csv('submission.csv', index=False, float_format='%.6f')