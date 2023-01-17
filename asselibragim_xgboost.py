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
dataset = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')

test_ds = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')

dataset.head()
x = dataset.iloc[:, 2:].values

y = dataset.iloc[:, 1].values
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 10, stratify = y)
from sklearn.utils.testing import ignore_warnings

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from xgboost import XGBClassifier
xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=1, gamma=1,

              learning_rate=0.1, max_delta_step=0, max_depth=8,

              min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,

              nthread=None, objective='binary:logistic', random_state=0,

              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

              silent=None, subsample=1, verbosity=1)

xgb.fit(x_train, y_train) 
test = test_ds.iloc[:, 1:].values

y_pred = xgb.predict(test)

result = pd.DataFrame({"target" : np.array(y_pred).T})

# train_ds['target'].value_counts()

result['target'].value_counts()

# print(classification_report(y_test, y_pred))
sub = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')

sub['target'] = y_pred

sub.to_csv('xgboost1.csv', index=False)
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'