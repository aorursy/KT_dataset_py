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
df_train = pd.read_csv('../input/train.csv', )

df_test = pd.read_csv('../input/test.csv')

df_train.tail()
df_train.drop(columns=['ID_code'], inplace=True)

df_test.drop(columns=['ID_code'], inplace=True)

print(df_train.shape, df_test.shape)
df_train.info()
print(df_train['target'].value_counts())
df_train.describe().T
df_test.tail()
X_feature = df_train.iloc[:, 1:]

y_label = df_train.iloc[:, 0]

print(X_feature.shape)
y_label.tail()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_feature, y_label,test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
from lightgbm import LGBMClassifier



lgbm_clf = LGBMClassifier(n_estimators=500)



evals = [(X_test, y_test)]

lgbm_clf.fit(X_train, y_train, early_stopping_rounds=100, eval_metric="auc", eval_set=evals, verbose=0)
from sklearn.metrics import roc_auc_score



lgbm_roc_score = roc_auc_score(y_test, lgbm_clf.predict_proba(X_test)[:, 1], average='macro')

print('ROC AUC: {0:.4f}'.format(lgbm_roc_score))
from sklearn.model_selection import GridSearchCV



LGBM_clf = LGBMClassifier(n_estimators=100)



params = {

    'num_leaves': [32],

    'max_depth': [128],

    'min_child_samples': [60, 100],

    'subsample': [0.8],

}



gridcv = GridSearchCV(lgbm_clf, param_grid=params, cv=5, verbose=0)

gridcv.fit(X_train, y_train, early_stopping_rounds=10, eval_metric='auc', eval_set=[

    (X_train, y_train), (X_test, y_test)])
print('최적 파라미터: ', gridcv.best_params_)
lgbm_roc_score = roc_auc_score(y_test, gridcv.predict_proba(X_test)[:, 1], average='macro')

print('ROC AUC: {0:.4f}'.format(lgbm_roc_score))
lgbm_clf = LGBMClassifier(

    n_estimators=1000, num_leaves=32,

    subsamples=0.8, min_child_samples=100,

    max_depth=128)



evals = [(X_test, y_test)]

lgbm_clf.fit(

    X_train, y_train, early_stopping_rounds=100,

    eval_metric="auc", eval_set=evals,

    verbose=False)
lgbm_roc_score = roc_auc_score(y_test, lgbm_clf.predict_proba(X_test)[:, 1], average='macro')

print('ROC AUC: {0:.4f}'.format(lgbm_roc_score))
sample_sub = pd.read_csv('../input/sample_submission.csv')

sample_sub.head()
df_test.head()
df_test2 = pd.read_csv('../input/test.csv')

df_test2.head()
submission = pd.DataFrame({

        "ID_code": df_test2["ID_code"],

        "target": lgbm_clf.predict(df_test)

    })

submission.to_csv('submission.csv', index=False)