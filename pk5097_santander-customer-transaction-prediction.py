# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")

sub=pd.read_csv("../input/sample_submission.csv")
train.head()
print(train.shape,test.shape)

test.head()
train.describe()
from sklearn.preprocessing import StandardScaler

stnd = StandardScaler()
x_train=train.drop(['ID_code','target'],axis=1)

y_train=train['target']

x_test=test.drop(['ID_code'],axis=1)
train.shape
data = stnd.fit_transform(x_train)

dtest = stnd.fit_transform(x_test)
'''import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import ShuffleSplit

import xgboost as xgb





for model_type in [LogisticRegression, KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier,AdaBoostClassifier, xgb.XGBClassifier]:

    clf = model_type()

    kfold = KFold(n_splits=5, shuffle=True)

    cv = ShuffleSplit(n_splits=100, test_size=0.3, random_state=0)

    scores = cross_val_score(clf, data, y_train, cv=cv, scoring='roc_auc')'''

    #print("{} Accuracy: {})".format(model_type.__name__, scores.mean()))
lr=LogisticRegression(solver='liblinear', class_weight='balanced', penalty='l1',C=0.1)
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(data,y_train, test_size=0.4, random_state=42)
lr.fit(X_train,Y_train)

from sklearn.metrics import roc_auc_score

roc_auc_score(Y_test , lr.predict(X_test))
clf = RandomForestClassifier(n_estimators=50, max_depth = 10,random_state=0)

clf.fit(X_train,Y_train)

roc_auc_score(Y_test , clf.predict(X_test))
pre=lr.predict(dtest)
sub.head()
sub["target"]=pre
sub.to_csv('sample_submission.csv',index=False)