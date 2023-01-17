# Importing required modules

import pandas as pd

from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from datetime import datetime

from sklearn.metrics import roc_auc_score

import numpy as np
# Importing datasets

data_train = pd.read_csv('../input/cat-in-the-dat/train.csv')

data_test = pd.read_csv('../input/cat-in-the-dat/test.csv')
# Checking Cardinality of various columns and their data types

for col in data_train.columns:

    print(col," --- ",len(data_train[col].value_counts()),"--- ",data_train[col].dtype)
# Checking the uniqe value to predict

print(data_train['target'].value_counts())
# Dropping target and id and select the categorical columns

y = data_train['target']

data_id = data_test['id']



data_train=data_train.drop(['id','target'],axis=1)

data_test=data_test.drop(['id'],axis=1)



cate_cols = [cols for cols in data_train.columns if data_train[cols].dtype == 'object']
# Label Encoding the categorical columns 

encoder = LabelEncoder()

for col in cate_cols:

    data_train[col] = pd.DataFrame(encoder.fit_transform(data_train[col]))

    data_test[col] = pd.DataFrame(encoder.fit_transform(data_test[col]))   

x_train,x_valid,y_train,y_valid = train_test_split(data_train,y,random_state=1)
# Describing our model with 200 estimators and 2 scaled position weight (For imbalanced data)

clf = XGBClassifier(n_estimators=200,scale_pos_weight=2,random_state=1,colsample_bytree=0.5)

clf.fit(x_train,y_train)
predictions = clf.predict_proba(x_valid)[:,1]
# Calculating the score using roc_auc_score

score = roc_auc_score(y_valid,predictions)

print(score)
predict = clf.predict_proba(data_test)[:,1]
submission = pd.DataFrame({'id': data_id, 'target': predict})

submission.to_csv('submission.csv', index=False)