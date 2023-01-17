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
df_train = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/train.csv")

df_test = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/test.csv")
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.metrics import roc_auc_score

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from category_encoders import *

from xgboost import XGBClassifier



X = df_train.drop(["id", "target"], axis=1)

y = df_train['target']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1993)

cols = list(X_train.columns)



encoder = TargetEncoder(cols=cols)

lr_est = LogisticRegression()



pipeline = Pipeline([

    ("Encoder", encoder),

    ("Estimator", lr_est)

])



params = {

    "Encoder__smoothing" : [0.1, 0.3, 0.9],

    "Encoder__min_samples_leaf" : [10,20,100],

}



search = GridSearchCV(pipeline, params, verbose=1,cv=StratifiedKFold(n_splits=5).get_n_splits([X_train, y_train]))

search.fit(X_train, y_train)

df_results = pd.DataFrame(search.cv_results_)

df_results
pipeline = Pipeline([

    ("Encoder", encoder),

    ("Estimator", lr_est)

])



params = {

    "Encoder__smoothing" : [0.1],

    "Encoder__min_samples_leaf" : [100],

    "Estimator__C": [0.1, 0.5, 1.0],

    "Estimator__solver": ["lbfgs", "liblinear"]

}



search = GridSearchCV(pipeline, params, verbose=1,cv=3)

search.fit(X_train, y_train)

df_results = pd.DataFrame(search.cv_results_)

df_results
final_pipeline = Pipeline([

    ("Encoder", TargetEncoder(cols=cols, smoothing=0.1, min_samples_leaf=100)),

    ("Estimator", LogisticRegression(solver="liblinear"))

])



final_pipeline.fit(X_train, y_train)

train_preds = final_pipeline.predict_proba(X_train)[:,1]

test_preds = final_pipeline.predict_proba(X_test)[:,1]



train_roc = roc_auc_score(y_train, train_preds)

test_roc = roc_auc_score(y_test, test_preds)

print("Train ROC = {:.3f}".format(train_roc))

print("Test ROC = {:.3f}".format(test_roc))

final_pipeline.fit(X, y)

final_preds = final_pipeline.predict_proba(df_test.drop('id', axis=1))[:,1]

df_preds = pd.DataFrame({'id': df_test['id'], 'target': final_preds})

df_preds.to_csv("LogReg_TargetEnc_Submission.csv", index=False)