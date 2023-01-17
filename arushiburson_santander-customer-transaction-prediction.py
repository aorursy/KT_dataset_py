#importing libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
data = pd.read_csv('../input/train.csv')

data.head()
data.info()
data.describe()
#checking target column

sns.countplot(data['target'])
#checking for null values

data.columns[data.isnull().any()]
#checking count for possible values of target

data.groupby('target').count()['ID_code']
y = data['target']

X = data.drop(['target', 'ID_code'], axis=1)



#scaling dataset

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(X)

X_scaled = sc.transform(X)

X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

X_scaled.head()
#splitting dataset into test and train

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=101, stratify=y)

X_train.head()
X_train.shape, X_test.shape
#using random forests

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

rfc.fit(X_train, y_train)

predictionRFC = rfc.predict_proba(X_test)

predictionRFC
#probability that a customer will make this transaction (target class 1)

prob = [1 - item[0] for item in predictionRFC] 

prob[:5]
#evaluating performance of the model

from sklearn.metrics import roc_auc_score

roc_auc_score(y_test, prob)
#loading test dataset

test = pd.read_csv('../input/test.csv')

test.head()
#making prediction on test dataset

test_ID = test['ID_code']

test_sc = test.drop(['ID_code'], axis=1)

test_scaled = sc.transform(test_sc)

#test_scaled = pd.DataFrame(test_scaled, columns = test_sc.columns)

predicted = pd.DataFrame([1 - item[0] for item in rfc.predict_proba(test_scaled)])

predicted.rename(columns={0: 'target'}, inplace=True)

predicted_output = pd.concat([test_ID, predicted['target']], axis=1)

predicted_output.head()
#resampling imbalanced dataset using SMOTE/ADASYN

#sm = SMOTE(random_state=12, ratio = 1.0)

#X_train_res, y_train_res = sm.fit_sample(X_train_scaled, y_train)

#type(X_train_res), type(y_train_res)
predicted_output.to_csv('PredictedRFC', index=False)