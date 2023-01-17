# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import warnings

warnings.filterwarnings("ignore")

import numpy as np

import pandas as pd

import sklearn

from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
cd /kaggle/input/eval-lab-2-f464
df = pd.read_csv("train.csv")
df.head()
df.isnull().isnull().sum()
from sklearn import tree
clf = tree.DecisionTreeClassifier()
x_val=df.drop(['class'], axis=1)
x_val.head()
y_val=pd.read_csv("train.csv")
y_val = y_val["class"].copy()
y_val.head()
clf = clf.fit(x_val, y_val)
test = pd.read_csv("test.csv")

test.head()
pred = clf.predict(test)
pred
sample = pd.read_csv("sample_submission.csv")
sample
submission = pd.DataFrame()

submission['id'] = test["id"].copy()
submission['class']=pred
submission.head()
!cd /kaggle/working

!pwd

submission.to_csv("/kaggle/working/output/2017A7PS0062G.csv",index=False)
from xgboost import XGBClassifier



xgb_clf = XGBClassifier().fit(x_val,y_val)

y_pred_xgb = xgb_clf.predict(test)
submission = pd.DataFrame()

submission['id'] = test["id"].copy()

submission['class']=pred

submission.to_csv("/kaggle/working/output/2017A7PS0062G.csv",index=False)
from sklearn.ensemble import GradientBoostingClassifier



gb_clf = GradientBoostingClassifier().fit(x_val,y_val)

y_pred_gb = gb_clf.predict(test)



submission = pd.DataFrame()

submission['id'] = test["id"].copy()

submission['class']=pred

submission.to_csv("/kaggle/working/output/2017A7PS0062G.csv",index=False)
df.head()
df.corr()
from sklearn import preprocessing



x = x_val.values

min_max_scaler = preprocessing.MinMaxScaler()

x_val_scaled = min_max_scaler.fit_transform(x)

x_val = pd.DataFrame(x_val_scaled)

x_val.head()
x=df.drop(['class'], axis=1)

x = x.columns

x_val.columns = x

x_val.head()
x = test.values

min_max_scaler = preprocessing.MinMaxScaler()

test_scaled = min_max_scaler.fit_transform(x)

test = pd.DataFrame(test_scaled)

test_columns = pd.read_csv("test.csv")

test_col = test_columns.columns

test.columns = test_col

test.head()
from sklearn.ensemble import GradientBoostingClassifier



gb_clf = GradientBoostingClassifier().fit(x_val,y_val)

y_pred_gb = gb_clf.predict(test)



submission = pd.DataFrame()

submission['id'] = test_columns["id"].copy()

submission['class']=pred

submission.to_csv("/kaggle/working/output/2017A7PS0062G.csv",index=False)
submission.head()