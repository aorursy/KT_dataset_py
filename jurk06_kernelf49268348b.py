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
train=pd.read_csv('../input/blood-train.csv')
test=pd.read_csv('../input/blood-test.csv')
sub=pd.read_csv('../input/blood-format.csv')
sub.head()
feat_col=['Made Donation in March 2007']
X=train.drop(feat_col, axis=1)
y=train.iloc[:,5]
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

gbrt = GradientBoostingClassifier(learning_rate=0.01, random_state=0)
gbrt.fit(X_train, y_train)
mena=gbrt.decision_function(X_test)
mena.mean()
pred=gbrt.predict_proba(X_test).mean()

pred
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train, y_train)
pred=lr.decision_function(X_test)
prd=lr.predict_proba(X_test).mean()
prd
submission = pd.DataFrame({"Unnamed: 0":test['Unnamed: 0'], "Made Donation in March 2007": prd})

submission.to_csv('submission_format.csv', index=False)
submission = pd.read_csv('submission_format.csv')
submission.head()
