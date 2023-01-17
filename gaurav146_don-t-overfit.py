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
import numpy as np

import pandas as pd

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

import xgboost as xgb

from sklearn.linear_model import ElasticNet

from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

sub = pd.read_csv("../input/sample_submission.csv")

sub['id'] = test['id']
train.head()
#train.groupby('target').count().plot(kind='barh', title='Target Distribution', figsize=(15, 5))

#plt.show()
train.describe()
train_X = train.drop(['id','target'], axis=1)

train_y = train['target']

test = test.drop(labels = 'id', axis = 1)
test.shape
test.head()
x = train.drop(labels = 'target', axis=1)

y = train["target"]
from sklearn.model_selection import train_test_split

import lightgbm as lgb

d_train = lgb.Dataset(train_X, label=train_y)



params = {}

params['alpha'] = 0.2

params['l1_ratio'] = 0.31

params['precompute'] = True

params['selection'] = 'random'

params['tol'] = 0.001 

params['random_state'] = 2

clf = lgb.train(params, d_train, 100)
sub["target"] = abs(clf.predict(test))
sub
#pred = rand_cv.predict(test)
#pred
#roc_auc_score(y_test, pre)
#pred = ad.predict(test)
#len(pred)
#test.shape


#sub['target'] = pred
sub.head(23)


sub.to_csv("sub.csv",index= False)

sub.head(23)