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
dataset_train = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')

dataset_test = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')
dataset_train.head()
train32 = dataset_train.drop(['ID_code', 'target'], axis = 1).astype('float32')
dataset_test.head()
dataset_test.describe()
X_train = dataset_train.iloc[:, 2:].values
X_train
y_train = dataset_train.target.values

y_train
X_test = dataset_test.values

X_test
X_train =dataset_train.iloc[:, 2:].values

Y_train = dataset_train.target.values

X_test = dataset_test.iloc[:, 1:].values
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(X_train,Y_train)
predict_values = logmodel.predict(X_test)
from sklearn.metrics import classification_report

print(classification_report(Y_train,predict_values))
submission_log = pd.DataFrame({'ID_code':dataset_test.ID_code.values})

submission_log['target'] = predict_values

submission_log.to_csv('submission_logreg.csv', index=False)
from sklearn.naive_bayes import GaussianNB

features = [x for x in dataset_train.columns if 'var_' in x]
nv = GaussianNB()
nv.fit(dataset_train[features], dataset_train['target'])
dataset_test['target'] = nv.predict_proba(dataset_test[features])[:, 1]
dataset_test[['ID_code', 'target']].to_csv('submission_GaussianNV.csv', index=False)
from sklearn.model_selection import StratifiedKFold

import xgboost as xgb
features = dataset_train.drop(['ID_code','target'], axis=1).values

targets = dataset_train.target.values
#Set params

params = {

    'min_child_weight': 10.0,

    'objective': 'binary:logistic',

    'max_depth': 7,

    'max_delta_step': 1.8,

    'colsample_bytree': 0.4,

    'subsample': 0.8,

    'eta': 0.025,

    'gamma': 0.65,

    'num_boost_round' : 700

    }
X = dataset_train.drop(['ID_code', 'target'], axis=1).values

y = dataset_train.target.values

test_id = dataset_test.ID_code.values

test = dataset_test.drop('ID_code', axis=1)
submission = pd.DataFrame()

submission['ID_code'] = test_id

submission['target'] = np.zeros_like(test_id)

submission.to_csv('submission_XGBoost.csv', index=False)

X_train = dataset_train.iloc[:, dataset_train.columns != 'target'].values

Y_train = dataset_train.iloc[:, 1].values

X_test = dataset_test.values
dataset_train.target.value_counts() 
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()





X_train[:,0] = le.fit_transform(X_train[:,0])

X_test[:,0] = le.fit_transform(X_test[:,0])

knn = KNeighborsClassifier(11)
knn.fit(X_train, Y_train)
y_preds = knn.predict(X_test)

pd.concat([test.ID_code, pd.Series(y_preds).rename('target')], axis = 1).to_csv('submission_knn_fix.csv', index =False)