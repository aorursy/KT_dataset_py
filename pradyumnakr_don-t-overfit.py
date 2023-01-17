# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import warnings

warnings.filterwarnings('ignore')





# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')
train.head()
train = train.set_index('id' , drop = True)

train.head()
test.head()
test=test.set_index('id',drop=True)
test.head()
y=train['target']

train=train.drop(['target'],axis=1)

train.info()

train.head()
test.info()


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

train=scaler.fit_transform(train)

test=scaler.transform(test)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV , train_test_split , cross_val_score

clf = LogisticRegression(class_weight='balanced', solver='liblinear', penalty ='l1', C= 0.1, max_iter=10000).fit(train, y)

scores = cross_val_score(clf, train, y, cv=5)

scores
clf.fit(train,y)

ans = clf.predict_proba(test)

ans
submit = pd.read_csv('../input/sample_submission.csv')

submit.columns
submit['target'] = ans[:,1]
submit.to_csv('submit.csv', index = False)
submit