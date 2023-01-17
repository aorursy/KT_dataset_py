# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import xgboost as xgb

train = pd.read_csv('../input/homesite-quote-conversion/train.csv.zip')
test = pd.read_csv('../input/homesite-quote-conversion/test.csv.zip')
train.head()
y = train.QuoteConversion_Flag.values
train = train.drop(['QuoteNumber','QuoteConversion_Flag'],axis = 1)
test = test.drop(['QuoteNumber'],axis = 1)
train['date'] = pd.to_datetime(pd.Series(train['Original_Quote_Date']))
train = train.drop(['Original_Quote_Date'],axis = 1)
test['date'] = pd.to_datetime(pd.Series(test['Original_Quote_Date']))
test = test.drop(['Original_Quote_Date'],axis = 1)
train['year'] = train['date'].apply(lambda x : int(str(x)[:4]))
train['month'] = train['date'].apply(lambda x: int(str(x)[5:7]))
train['day'] = train['date'].dt.dayofweek
test['year'] = test['date'].apply(lambda x : int(str(x)[:4]))
test['month'] = test['date'].apply(lambda x: int(str(x)[5:7]))
test['day'] = test['date'].dt.dayofweek
train = train.drop('date',axis=1)
test = test.drop('date',axis =1)
train = train.fillna(-1)
test = test.fillna(-1)
for f in train.columns:
    if train[f].dtype=='object':
        print(f)
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[f].values) + list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))
clf = xgb.XGBClassifier(n_estimator=100,
                       nthread =-1,
                       max_depth =9,
                       learning_rate = 0.026,
                       silent = True,
                       subsample = 0.8,
                       colsample_bytree=0.75)
xgb_model = clf.fit(train,y, eval_metric="auc")
preds = clf.predict_proba(test)[:,1]
sample = pd.read_csv('../input/homesite-quote-conversion/sample_submission.csv.zip')
sample.QuoteConversion_Flag = preds
sample.to_csv('xgb_benchmark.csv',index=False) 