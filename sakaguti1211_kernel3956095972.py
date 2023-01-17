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

import seaborn as sns

import sklearn

import matplotlib.pyplot as plt

import os

from pandas import Series,DataFrame

from matplotlib.pyplot import scatter

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.ensemble import BaggingClassifier

import optuna
train=pd.read_csv("../input/train.csv")

Y=train['target']

X=train.drop(['id','target'],axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.1,random_state=3)
from sklearn.linear_model import LogisticRegression

log_model1=BaggingClassifier(base_estimator=LogisticRegression(class_weight='balanced',penalty='l1', C=0.1, solver='liblinear'),n_estimators=153,random_state=54)

log_model1.fit(X_train,Y_train)

ans1=log_model1.predict(X_test)

print(accuracy_score(Y_test,ans1))
test=pd.read_csv("../input/test.csv")



pX=test.drop(['id'],axis=1)
class_ans=log_model1.predict(pX)
test['target']=class_ans

test[["id",'target']].head()
test[["id",'target']].to_csv("submission.csv",index=False)