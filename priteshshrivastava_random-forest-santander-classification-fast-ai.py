!pip install --upgrade pip

!pip install fastai==0.7.0 ## Based on Fast.ai ML course
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from IPython.display import display

from fastai.imports import *

from fastai.structured import *

from pandas_summary import DataFrameSummary

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.model_selection import train_test_split

import os

print(os.listdir("../input/"))
train = pd.read_csv("../input/santander-customer-transaction-prediction/train.csv")

test = pd.read_csv("../input/santander-customer-transaction-prediction/test.csv")
train.describe(include='all')
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler()

df_resampled, y_resampled = ros.fit_sample(train, train['target'])

df_resampled = pd.DataFrame(df_resampled, columns = train.columns)

train['target'].mean(), df_resampled['target'].mean()

train_cats(df_resampled)

apply_cats(test, df_resampled)
df_trn, y_trn, nas = proc_df(df_resampled, 'target')

df_test, _, _ = proc_df(test, na_dict=nas)
X_train, X_valid, y_train, y_valid = train_test_split(df_trn, y_trn, test_size=0.33, random_state=42)
from sklearn.metrics import roc_auc_score



def print_score(m):

    res = [roc_auc_score(m.predict(X_train), y_train), roc_auc_score(m.predict(X_valid), y_valid)]

    print(res)
set_rf_samples(100000)  ## To train faster, we can train on a smaller subset

m = RandomForestClassifier(n_jobs=-1, n_estimators = 80, max_depth = 10, min_samples_leaf = 10, min_samples_split = 10)

%time m.fit(X_train, y_train)
%time print_score(m)
pred = m.predict(df_test)
submission = pd.read_csv('../input/santander-customer-transaction-prediction/sample_submission.csv')

submission['target'] = pred

submission.to_csv('rf_submission_iter3.csv', index=False)