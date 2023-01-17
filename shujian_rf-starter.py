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
from sklearn.ensemble import RandomForestClassifier
import gc
!wc -l ../input/train.csv
!wc -l ../input/test.csv
train = pd.read_csv("../input/train.csv", nrows = 10000000)
# train = pd.read_csv("../input/train_sample.csv")

train.head()
test = pd.read_csv("../input/test.csv")
test.head()
sub = pd.read_csv("../input/sample_submission.csv")
sub.head()
y_train = train['is_attributed']
# x_train = train[['ip', 'app', 'device', 'os', 'channel', 'click_time']]
# x_test  = test[['ip', 'app', 'device', 'os', 'channel', 'click_time']]
x_train = train[['ip', 'app', 'device', 'os', 'channel']]
x_test  = test[['ip', 'app', 'device', 'os', 'channel']]

del train
del test
gc.collect()
clf = RandomForestClassifier(max_depth=6, random_state=0)
clf.fit(x_train, y_train)
# y_pred = clf.predict(x_test)
y_pred = clf.predict_proba(x_test)
sub['is_attributed'] = y_pred[:,1]
sub.head()
sub.to_csv('sub_rf.csv', index=False)