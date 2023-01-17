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
train_t=pd.read_csv("../input/mydata/trainplus.csv")

tY=train_t['target']

tX=train_t.drop(['id','target','sum'],axis=1)



train_p=pd.read_csv("../input/mydata/traintest.csv")

tpY=train_p['target']

tpX=train_p.drop(['id','target','sum'],axis=1)
from sklearn.linear_model import LogisticRegression



log_model1=BaggingClassifier(base_estimator=LogisticRegression(class_weight='balanced',penalty='l1', C=0.1, solver='liblinear'),

                                 n_estimators=270,random_state=54)

log_model1.fit(tX,tY)

ans1=log_model1.predict(tpX)



print(accuracy_score(tpY,ans1))
test=pd.read_csv('../input/dont-overfit-ii/test.csv')

ppX=test.drop(['id'],axis=1)
ppX.head()
ans1=log_model1.predict(ppX)
ans1
test['target']=ans1

test[["id",'target']].head()
test[["id",'target']].to_csv("submission.csv",index=False)