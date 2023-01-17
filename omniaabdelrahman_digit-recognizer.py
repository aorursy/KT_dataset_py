# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import svm

from sklearn.linear_model import LogisticRegression

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test_data =  pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
train_data.shape
label=train_data['label']

train_data=train_data.drop('label', axis=1)
train_data.shape
train_data[train_data>0]=1
train_data.head()
''''from sklearn import decomposition

from sklearn import datasets

pca = decomposition.PCA(n_components=100) 

pca.fit(train_data)

train_data.shape'''
svc = svm.SVC()

svc.fit(train_data, label.values.ravel())

svc.score(train_data,label)
''''logobj = LogisticRegression()

logobj.fit(train_data,label)

y_pred = logobj.predict(test_data)

log = round(logobj.score(train_data, label)*100 , 1)

log'''
submission = pd.read_csv('../input/digit-recognizer/sample_submission.csv')

submission.to_csv('result.csv',index=False)