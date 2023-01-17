# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import Imputer

#用来补全缺失值

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
try_train = pd.read_csv('../input/train.csv')

try_test = pd.read_csv('../input/test.csv')

try_train.head()
try_train.describe()
try_train.hist(figsize=(20,20))

plt.figure()
bins=[0,10,20,30,40,50,60,70,80,90,100]

factor = pd.cut(try_train['Age'],bins = bins)

sns.pointplot(x=factor,y='Survived',data=try_train)
try_train[['Parch','Survived']].groupby(['Parch']).mean().plot.bar()
try_train[['SibSp','Survived']].groupby(['SibSp']).mean().plot.bar()
try_train[['Pclass','Survived']].groupby(['Pclass']).mean().plot.bar()
try_train[['Sex','Survived']].groupby(['Sex']).mean().plot.bar()
try_train[['Embarked','Survived']].groupby(['Embarked']).mean().plot.bar()
try_train.Age=try_train.Age.fillna(try_train.Age.median())
try_train.describe()
try_train.head()

embark_dummies  = pd.get_dummies(try_train['Embarked'])

try_train = try_train.join(embark_dummies)

embark_dummies = try_train[['S', 'C', 'Q']]

sex_dummies  = pd.get_dummies(try_train['Sex'])

try_train = try_train.join(sex_dummies)

sex_dummies = try_train[['male', 'female']]
try_train.drop(['Sex'],axis=1,inplace=True)

try_train.drop(['Embarked'],axis=1,inplace=True)

try_train.head()

features = ['Pclass','Age','SibSp','Parch','Fare','male','female','S','Q','C']

alg = LogisticRegression()

kf = KFold(n_splits=5, random_state=1)

predictions = list()

for train, test in kf.split(try_train):

    k_train = try_train[features].iloc[train,:]

    k_label = try_train.Survived.iloc[train]

    alg.fit(k_train,k_label)

    k_predictions = alg.predict(try_train[features].iloc[test,:])

    predictions.append(k_predictions)



predictions = np.concatenate(predictions,axis=0)

accuracy_score(try_train.Survived,predictions)

try_test.head(10)
try_test.Age=try_test.Age.fillna(try_test.Age.median())

embark_dummies  = pd.get_dummies(try_test['Embarked'])

try_test = try_test.join(embark_dummies)

embark_dummies = try_test[['S', 'C', 'Q']]

sex_dummies  = pd.get_dummies(try_test['Sex'])

try_test = try_test.join(sex_dummies)

sex_dummies = try_test[['male', 'female']]

try_test.drop(['Embarked','Sex'], axis=1,inplace=True)
try_test.head()
try_test[features] = Imputer().fit_transform(try_test[features])

alg = LogisticRegression()

kf = KFold(n_splits=5,random_state=1)

for train, test in kf.split(try_train):

    k_train = try_train[features].iloc[train,:]

    k_label = try_train.Survived.iloc[train]

    alg.fit(k_train,k_label)

predictions = alg.predict(try_test[features])
_try = pd.DataFrame([try_test.PassengerId,pd.Series(predictions)],index=['PassengerId','Survived'])

_try.to_csv('gender_submission.csv',index=False)

try_prediction=pd.read_csv('gender_submission.csv')
try_prediction.head()