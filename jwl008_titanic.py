# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

test_data.head()
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

train_data.head()
train_data.info()
test_data.info()
import matplotlib.pyplot as plt

import seaborn as sns
numeric_features = ['Survived','Age','SibSp','Parch','Fare']

pd.plotting.scatter_matrix(train_data[numeric_features],figsize = (10,10))

plt.show()
train_data['Pclass'] = pd.Categorical(train_data['Pclass'])

cat_features = ['Pclass','Sex','Embarked']

fig, ax = plt.subplots(1,3,figsize = (25,5))

for i,feature in enumerate(cat_features):

    sns.barplot(x = 'Survived', y = feature, data = train_data,ax = ax[i],ci = None)
train_data[train_data.Embarked.isna()]
train_data = train_data.drop(train_data[train_data.Embarked.isna()].index)
train_data.info()
test_data[test_data.Fare.isna()]
test_data[(test_data.Pclass == 3) & (test_data.Embarked == 'S')].Fare.mean()
test_data.loc[test_data[test_data.Fare.isna()].index, 'Fare'] = test_data[(test_data.Pclass == 3) & (test_data.Embarked == 'S')].Fare.mean()

test_data['Pclass'] = pd.Categorical(test_data['Pclass'])

test_data.info()
feature = ['Pclass','Sex','Embarked','SibSp','Parch','Fare']

train_x = train_data[feature]

train_x = pd.get_dummies(train_x)

train_x.info()
train_x = train_x.to_numpy()

train_y = train_data['Survived'].to_numpy()
test_x = pd.get_dummies(test_data[feature])

test_x.info()

test_x = test_x.to_numpy()
from sklearn.linear_model import LogisticRegression

logit = LogisticRegression(max_iter = 10000).fit(train_x, train_y)

logit_pre = logit.predict(test_x)

logit_out = pd.DataFrame({'PassengerId':test_data.PassengerId, 'Survived': logit_pre})

logit_out.to_csv('result_logit.csv',index = False)
from sklearn import svm

sc = svm.SVC().fit(train_x,train_y)

sc_pre = sc.predict(test_x)

sc_out = pd.DataFrame({'PassengerId':test_data.PassengerId, 'Survived': sc_pre})

sc_out.to_csv('result_svm.csv',index = False)
from sklearn.naive_bayes import GaussianNB

gnb_pre = GaussianNB().fit(train_x,train_y).predict(test_x)

gnb_out = pd.DataFrame({'PassengerId':test_data.PassengerId, 'Survived': gnb_pre})

gnb_out.to_csv('result_naivebayes.csv',index = False)
from sklearn import tree

dtc_pre = tree.DecisionTreeClassifier().fit(train_x,train_y).predict(test_x)

dtc_out = pd.DataFrame({'PassengerId':test_data.PassengerId, 'Survived': dtc_pre})

dtc_out.to_csv('result_decisiontree.csv',index = False)
from sklearn.ensemble import RandomForestClassifier

rfc_pre = RandomForestClassifier().fit(train_x,train_y).predict(test_x)

rfc_out = pd.DataFrame({'PassengerId':test_data.PassengerId, 'Survived': rfc_pre})

rfc_out.to_csv('result_randomforest.csv',index = False)