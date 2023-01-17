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
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()
print(train.shape,test.shape)
test.head()
from sklearn.preprocessing import StandardScaler

stnd = StandardScaler()
x_train=train.drop(['id','target'],axis=1)

y_train=train['target']

x_test=test.drop(['id'],axis=1)

train=stnd.fit_transform(x_train)

test=stnd.fit_transform(x_test)

from sklearn.linear_model import LogisticRegression



lr =LogisticRegression(class_weight='balanced')
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import uniform

clf = LogisticRegression(class_weight='balanced', solver='liblinear')

penalty = ['l1', 'l2']

C = uniform(loc=0, scale=4)

hyperparameters = dict(C=C, penalty=penalty)

rand_cv = RandomizedSearchCV(clf, hyperparameters, random_state=1, n_iter=100, cv=5, verbose=0, n_jobs=-1, scoring='roc_auc')

best_model=rand_cv.fit(x_train,y_train)

print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])

print('Best C:', best_model.best_estimator_.get_params()['C'])

print('Best Score: {}'.format(best_model.best_score_))
lr =LogisticRegression(class_weight='balanced', penalty='l1',solver='liblinear',C = 0.17)
from sklearn.feature_selection import RFE

selector = RFE( lr , 25)

selector.fit(train, y_train)
sub=pd.read_csv("../input/sample_submission.csv")
pre=selector.predict(test)
sub['target']=pre
sub.head(10)
sub.to_csv('sample_submission.csv',index=False)