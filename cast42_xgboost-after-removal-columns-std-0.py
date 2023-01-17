# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
%matplotlib inline

pd.set_option('display.precision', 5)

training = pd.read_csv("../input/train.csv", index_col=0)
test = pd.read_csv("../input/test.csv", index_col=0)

print(training.shape)
print(test.shape)

X = training.iloc[:,:-1]
y = training.TARGET
import xgboost as xgb
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1301)
clf = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)
clf.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="auc",
        eval_set=[(X_test, y_test)])
        
y_pred = clf.predict(test)
y_pred = clf.predict_proba(test)
y_pred[:,1]
# Xgboost 
params = {"objective": "binary:logistic", "booster": "gbtree", "eval_metric":"auc"}

train_xgb = xgb.DMatrix(X, y)
test_xgb  = xgb.DMatrix(test)

gbm = xgb.train(params, train_xgb, 20)
y_pred = gbm.predict(test_xgb)
test.index
submission = pd.DataFrame({"ID":test.index, "TARGET":y_pred})
submission.to_csv("submission.csv", index=False)
remove = []
for col in training.columns:
    if training[col].std() == 0:
        remove.append(col)
training.drop(remove, axis=1, inplace=True)
test.drop(remove, axis=1, inplace=True)
X = training.iloc[:,:-1]
y = training.TARGET
# Xgboost 
params = {"objective": "binary:logistic", "booster": "gbtree", "eval_metric":"auc"}

train_xgb = xgb.DMatrix(X, y)
test_xgb  = xgb.DMatrix(test)

gbm = xgb.train(params, train_xgb, 20)
y_pred = gbm.predict(test_xgb)
submission = pd.DataFrame({"ID":test.index, "TARGET":y_pred})
submission.to_csv("submission.csv", index=False)
