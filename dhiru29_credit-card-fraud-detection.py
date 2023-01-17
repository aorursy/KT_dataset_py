# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
credit = pd.read_csv('../input/creditcard.csv')
# Any results you write to the current directory are saved as output.
credit.head()
credit.describe()
credit['Class'].value_counts()
# Now more Describe the data
import pandas_profiling as pf
pf.ProfileReport(credit)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import f1_score,accuracy_score
credit_dummy = pd.get_dummies(credit)
credit_dummy.shape
train, test = train_test_split(credit_dummy,test_size=.33,random_state=100)
train_y = train['Class']
test_y =test['Class']

train_x = train.drop('Class',axis=1)
test_x = test.drop('Class',axis=1)
rf = RandomForestClassifier()
rf.fit(train_x,train_y)

rf_pred = rf.predict(test_x)

rf_acc = accuracy_score(test_y,rf_pred)
rf_acc
gr = GradientBoostingClassifier()
gr.fit(train_x,train_y)

gr_pred = gr.predict(test_x)

gr_acc = accuracy_score(test_y,gr_pred)
gr_acc
import xgboost as xg
from xgboost import XGBClassifier
xg = xg.XGBClassifier()
xg.fit(train_x,train_y)

xg_pred = xg.predict(test_x)
xg_accura = accuracy_score(test_y,xg_pred)
xg_accura
