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
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

sample = pd.read_csv("../input/sample_submission.csv")
cv_y = train_df["target"]
train_df.head()
train_df = train_df.drop("ID_code", axis = 1)
cv_x = train_df.drop("target", axis = 1)
cv_x.head()
cv_y.head()
from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split
SEED = 1

X_train, X_test, y_train, y_test = train_test_split(cv_x, cv_y, test_size=0.3, stratify=cv_y, random_state=SEED)
dt = DecisionTreeClassifier(max_depth=1, random_state=SEED)

adb_clf = AdaBoostClassifier(base_estimator=dt, n_estimators=100)

adb_clf.fit(X_train, y_train)
y_pred_proba = adb_clf.predict_proba(X_test)[:,1]

adb_clf_roc_auc_score = roc_auc_score(y_test, y_pred_proba)

print('ROC AUC score: {:.2f}'.format(adb_clf_roc_auc_score))
test_final = test_df.drop("ID_code", axis = 1)
y_pred_prob_final = adb_clf.predict_proba(test_final)[:,1]
y_pred_prob_final[: 10]
ID_code = sample["ID_code"]
prediction = pd.DataFrame(y_pred_prob_final, index= ID_code)
prediction.columns = ["target"]

prediction.index.name = "ID_code"

prediction.head()
prediction.to_csv("prediction_santander_Decision_tree.csv")