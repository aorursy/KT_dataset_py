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
train = pd.read_csv('../input/train.csv',index_col=0)
#1

train.default.mean()
#2

train.groupby('ZIP')['default'].mean().idxmax()
#3 

train[train.year == 0].default.mean()
#4 

train.age.corr(train.income)
#5

from sklearn.ensemble import RandomForestClassifier

predictors = ['ZIP', 'rent', 'education', 'income', 'loan_size', 'payment_timing', 'job_stability','occupation']

target = ['default']

X = pd.get_dummies(train[predictors])

y = train[target]

clf = RandomForestClassifier(n_jobs=-1,n_estimators=100,oob_score=True,random_state=42)

clf.fit(X=X,y=y)
from sklearn.metrics import accuracy_score

accuracy_score(y_pred = clf.predict(X),y_true=y)

#6

clf.oob_score_
#7

test = pd.read_csv('../input/test.csv')

X_test = pd.get_dummies(test[predictors])

y_test = test[target].values

accuracy_score(y_pred = clf.predict(X_test),y_true=y_test)
#8

maj_dist = clf.predict(X_test[test.minority == 0])

maj_dist.mean()
#9

min_dist = clf.predict(X_test[test.minority == 1])

min_dist.mean()
#11

pred = clf.predict(X_test)
print(f'Share of succesful applicants that are: minority members: {test[~pred].minority.mean()*100:.2f}%,\ female: {test[~pred].sex.mean()*100:.2f}%')
print(f'Share of rejected applicants that are: minority members: {test[pred].minority.mean()*100:.2f}%,\ female: {test[pred].sex.mean()*100:.2f}%')
print(f'Share defaulting on their loans: minority members: {y_test[(test.minority == 1) & (~pred)].mean()*100:.2f}% \ non-minority: {y_test[(test.minority == 0) & (~pred)].mean()*100:.2f}% \ male: {y_test[(test.sex == 1) & (~pred)].mean()*100:.2f}% \ female: {y_test[(test.sex == 0) & (~pred)].mean()*100:.2f}%')