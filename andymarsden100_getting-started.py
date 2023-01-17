# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from collections import Counter

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# load in the sample data
train = pd.read_csv("../input/train_sample.csv")
# let's take a look
train.head()
# How imbalanced is the dataset
print(Counter(train['is_attributed']))

# Let's split the time data into separate columns
#train['year'] = pd.DatetimeIndex(train['click_time']).year
#train['month'] = pd.DatetimeIndex(train['click_time']).month
#train['day'] = pd.DatetimeIndex(train['click_time']).day
#train['hour'] = pd.DatetimeIndex(train['click_time']).hour
#train['minute'] = pd.DatetimeIndex(train['click_time']).minute
#train['second'] = pd.DatetimeIndex(train['click_time']).second
# let's drop the time column, and I think attribuited_time is only there if attributed is true - so drop this too
train = train.drop(['click_time','attributed_time'],1)
# Let's check it
train.head()
# Year and month are constants, so get rid and split into X and y
X = train.drop(['is_attributed'],1)
y = train['is_attributed']
# Create a test train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# Let's try random forests first
clf = RandomForestClassifier(n_estimators=500 , max_depth=20)
clf.fit(X_train,y_train)
preds = clf.predict(X_test)
roc_auc_score(y_test, preds)
# OK, let's try XGBoost!
import xgboost as xgb
clf = xgb.XGBClassifier(n_estimators=500,max_depth=20,learning_rate=0.01)
clf.fit(X_train,y_train)
preds = clf.predict(X_test)
roc_auc_score(y_test, preds)
