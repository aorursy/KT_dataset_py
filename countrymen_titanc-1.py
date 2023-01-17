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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
train.info()
train.describe()
sns.countplot(train['Survived'])
train['Survived'].value_counts()
train_corr = train.drop(['PassengerId'], axis=1).corr()

train_corr
plt.subplots(figsize=(15,9))

sns.heatmap(train_corr, annot=True, vmax=1, vmin=-1)
train.head()
train.groupby(by='Pclass')['Pclass', 'Survived'].mean()
train[['Pclass', 'Survived']].groupby('Pclass').mean().plot.bar()
train.groupby('Sex')['Sex','Survived'].mean()
train.groupby('Sex')['Sex','Survived'].mean().plot.bar()
train[['SibSp', 'Survived']].groupby('SibSp').mean().plot.bar()
train[['Parch', 'Survived']].groupby('Parch').mean().plot.bar()
train.groupby(['Age'])['Survived'].mean().plot()
sns.countplot('Embarked', hue='Survived', data=train)
test['Survived'] = 0
train_test = train.append(test)

train_test
train_test = pd.get_dummies(train_test,columns=['Pclass'])

train_test = pd.get_dummies(train_test,columns=["Sex"])

train_test = pd.get_dummies(train_test,columns=["Embarked"])
train_test.drop('Cabin', axis=1, inplace=True)
train_test.drop('Name', axis=1, inplace=True)

train_test.drop('Ticket', axis=1, inplace=True)

train_test
train_test.info()
train_test.loc[train_test["Fare"].isnull()]
train.groupby(by=["Pclass","Embarked"]).Fare.mean()
train_test["Fare"].fillna(14.644083, inplace=True)

train_test
train_test.info()
missing_age = train_test.drop(['Survived'],axis=1)

missing_age_train = missing_age[missing_age['Age'].notnull()]

missing_age_test = missing_age[missing_age['Age'].isnull()]
missing_age_X_train = missing_age_train.drop(['Age'], axis=1)

missing_age_Y_train = missing_age_train['Age']

missing_age_X_test = missing_age_test.drop(['Age'], axis=1)
# 先将数据标准化

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

#用测试集训练并标准化

ss.fit(missing_age_X_train)

missing_age_X_train = ss.transform(missing_age_X_train)

missing_age_X_test = ss.transform(missing_age_X_test)
from sklearn import linear_model

lin = linear_model.BayesianRidge()



lin.fit(missing_age_X_train,missing_age_Y_train)
train_test.loc[(train_test['Age'].isnull()), 'Age'] = lin.predict(missing_age_X_test)
train_test.info()
train_data = train_test[:891]

test_data = train_test[891:]

train_data_X = train_data.drop(['Survived'],axis=1)

train_data_Y = train_data['Survived']

test_data_X = test_data.drop(['Survived'],axis=1)
from sklearn.preprocessing import StandardScaler



ss2 = StandardScaler()

ss2.fit(train_data_X)

train_data_X_sd = ss2.transform(train_data_X)

test_data_X_sd = ss2.transform(test_data_X)
train_data_X_sd
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators=150,min_samples_leaf=3,max_depth=6,oob_score=True)

rf.fit(train_data_X,train_data_Y)



test["Survived"] = rf.predict(test_data_X)

RF = test[['PassengerId','Survived']].set_index('PassengerId')

RF.to_csv('RF.csv')
pd.read_csv('RF.csv')
import xgboost as xgb



xgb_model = xgb.XGBClassifier(n_estimators=150,min_samples_leaf=3,max_depth=6)

xgb_model.fit(train_data_X,train_data_Y)



test["Survived"] = xgb_model.predict(test_data_X)

XGB = test[['PassengerId','Survived']].set_index('PassengerId')

XGB.to_csv('XGB5.csv')
from sklearn.ensemble import VotingClassifier



from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=0.1,max_iter=100)



import xgboost as xgb

xgb_model = xgb.XGBClassifier(max_depth=6,min_samples_leaf=2,n_estimators=100,num_round = 5)



from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=200,min_samples_leaf=2,max_depth=6,oob_score=True)



from sklearn.ensemble import GradientBoostingClassifier

gbdt = GradientBoostingClassifier(learning_rate=0.1,min_samples_leaf=2,max_depth=6,n_estimators=100)



vot = VotingClassifier(estimators=[('lr', lr), ('rf', rf),('gbdt',gbdt),('xgb',xgb_model)], voting='hard')  # soft

vot.fit(train_data_X_sd,train_data_Y)



test["Survived"] = vot.predict(test_data_X_sd)

test[['PassengerId','Survived']].set_index('PassengerId').to_csv('vot5.csv')
pd.read_csv('vot5.csv')
!ls
from sklearn.linear_model import LogisticRegression

from sklearn import svm

import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier
X = train_data_X_sd

X_predict = test_data_X_sd

y = train_data_Y
clfs = [LogisticRegression(C=0.1,max_iter=100),

        xgb.XGBClassifier(max_depth=6,n_estimators=100,num_round = 5),

        RandomForestClassifier(n_estimators=100,max_depth=6,oob_score=True),

        GradientBoostingClassifier(learning_rate=0.3,max_depth=6,n_estimators=100)]
from mlxtend.classifier import StackingClassifier





sclf = StackingClassifier(classifiers=clfs, 

                          meta_classifier=LogisticRegression(C=0.1,max_iter=100))

sclf.fit(X, y)
sclf.predict(X_predict)
test = pd.read_csv("../input/test.csv")

test["Survived"] = sclf.predict(X_predict)

test[['PassengerId','Survived']].set_index('PassengerId').to_csv('stack3.csv')
!ls