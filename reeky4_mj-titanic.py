# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import sys

import scipy as sp

import sklearn

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

import random

import time
from sklearn import svm, tree, linear_model

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics
data_raw_train= pd.read_csv('/kaggle/input/titanic/train.csv', index_col='PassengerId')

data_test= pd.read_csv('/kaggle/input/titanic/test.csv',index_col='PassengerId')

labels_test= pd.read_csv('/kaggle/input/titanic/gender_submission.csv',index_col='PassengerId')
data_raw_train.describe(include='all')
data_raw_train.head()
data_raw_test=pd.concat([data_test, labels_test], axis=1)
data_raw_test.describe()
data_drop_train = data_raw_train.drop(['Name', 'Ticket', 'Cabin', 'Age'], axis = 1)

data_drop_train.head()
data_drop_test = data_raw_test.drop(['Name', 'Ticket', 'Cabin', 'Age'], axis = 1)

data_drop_test
data_drop_train.Embarked.fillna("U", inplace=True)

data_drop_test.Embarked.fillna("U", inplace=True)

data_drop_test.Fare.fillna(0, inplace=True)
#data_drop_test.dropna(inplace=True)

data_drop_train.dropna(inplace=True)
plt.figure(figsize=(12,12))

sns.pairplot(data_drop_train, hue='Survived', vars=['Pclass', 'Fare', 'Parch', 'SibSp'])

plt.show()
#_, bins = np.histogram(data_drop_train["Pclass"])

#g = sns.FacetGrid(data_drop_train, hue="Survived")

#g = g.map(sns.distplot, "Pclass", bins=bins)

#plt.show()
data_train=pd.get_dummies(data_drop_train, drop_first=True)

data_test=pd.get_dummies(data_drop_test, drop_first=True)
data_train.describe()
data_test['Embarked']=np.zeros(418)

data_test.describe()
y_train=np.array(data_train.Survived)

y_test=np.array(data_test.Survived)

y_train.reshape(-1,1)

y_test.reshape(-1,1)
x_train=data_train.drop(['Survived'],axis=1)

x_test=data_test.drop(['Survived'],axis=1)
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
scl=StandardScaler()

x_train=scl.fit_transform(x_train)

x_test=scl.fit_transform(x_test)
clf=SVC(kernel='linear')

clf.fit(x_train,y_train)
pred = clf.predict(x_test)

res = pd.DataFrame({'PassengerId': data_test.index, 'Survived': pred})

res
clf.score(x_test,y_test)
res.to_csv('/kaggle/working/titanic_sub1.csv', index=False)
#end of code