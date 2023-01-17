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
import pandas as pd
import numpy as np
data = pd.read_csv('../input/train.csv',nrows=1 * (10**7))
test = pd.read_csv('../input/test.csv')
data['click_time'] = pd.to_datetime(data['click_time'])
data['hour'] = data['click_time'].apply(lambda x: x.hour)
data['weekday'] = data['click_time'].apply(lambda x: x.weekday())
data['day'] = data['click_time'].apply(lambda x: x.day)
test['click_time'] = pd.to_datetime(test['click_time'])
test['hour'] = test['click_time'].apply(lambda x: x.hour)
test['weekday'] = test['click_time'].apply(lambda x: x.weekday())
test['day'] = test['click_time'].apply(lambda x: x.day)

import xgboost as xgb
from sklearn.model_selection import train_test_split
pars = {'eta' : 0.3,
       'max_depth' : 6,
       'objective' : 'binary:logistic',
       'eval_metric' : 'auc',
       'random_state' : 42}

x_train, x_test, y_train, y_test = train_test_split(data.drop(['attributed_time','is_attributed','click_time'],axis=1),
                                                    data['is_attributed'],test_size=0.3,random_state=42)
ls = [(xgb.DMatrix(x_train,y_train), 'train'), (xgb.DMatrix(x_test,y_test),'valid')]
report = {}
clf = xgb.train(pars, xgb.DMatrix(x_train,y_train),250,ls,maximize=True,evals_result=report)

pred = clf.predict(xgb.DMatrix(test.drop(['click_id','click_time'],axis=1)),ntree_limit=clf.best_ntree_limit)
sub = pd.DataFrame({'click_id' : test.click_id.values, 'is_attributed': pred})
sub.to_csv('submission.csv',index=False)

