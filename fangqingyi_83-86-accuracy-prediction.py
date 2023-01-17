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
#引入库

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')

#读取数据，合并一个数据集

train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

fullData = pd.concat(objs = [train, test], axis = 0).reset_index(drop = True)

#输出数据概要

print(train.info())
#查看数据缺失值

print(train.isnull().sum())

print(test.isnull().sum())
#检查Pclass  可视化

print(train[['Survived', 'Pclass']].groupby(['Pclass'], as_index = False).mean())

sns.barplot(x = 'Pclass', y = 'Survived', data = train)
#Sex

print(train[['Survived', 'Sex']].groupby(['Sex'], as_index = False).mean())

fig, ax = plt.subplots(1, 2, figsize=(8, 4))

sns.barplot(x='Sex', y='Survived', data=train, ax=ax[0])

ax[0].set_title('ratio')

sns.countplot(x = 'Sex', hue='Survived', data = train, ax = ax[1])

ax[1].set_title('count')
#SibSp

print(train[['Survived', 'SibSp']].groupby(['SibSp'], as_index = False).mean())

sns.barplot(x='SibSp', y='Survived', data=train)
#Parch

print(train[['Survived', 'Parch']].groupby(['Parch'], as_index=False).mean())

sns.barplot(x = 'Parch', y = 'Survived', data = train)
#Embaked  

print(train[['Survived','Embarked']].groupby(['Embarked'], as_index = False).mean())
#Fare

g = sns.kdeplot(train['Fare'][train['Survived']==1],color='Green',shade=True)

g = sns.kdeplot(train['Fare'][train['Survived']==0],color='Red',ax=g,shade=True)

g.set_xlabel('Fare')

g = g.legend(['S-Fare','Not-S-Fare'])
#Age

g = sns.FacetGrid(train, col = 'Survived')

g = g.map(sns.distplot, 'Age')
#清理数据，填补所有缺失值

train['Age'][train['Age'].isnull()] = np.random.normal(loc=fullData['Age'].mean(),scale=fullData['Age'].std(),size=[177,1])

test['Age'][test['Age'].isnull()] = np.random.normal(loc=fullData['Age'].mean(),scale=fullData['Age'].std(),size=[86,1])

test['Fare'] = test['Fare'].fillna(fullData['Fare'].mean(),inplace = False)

train['Embarked'] = train['Embarked'].fillna(fullData['Embarked'].describe().top,inplace=False)
#检查是否还有缺失

print(train.isnull().sum())

print(test.isnull().sum())
#map Embarked 

print(fullData['Embarked'].unique())

train['Embarked'] = train['Embarked'].map({'S':1,'C':2,'Q':3})

test['Embarked'] = test['Embarked'].map({'S':1,'C':2,'Q':3})
#map Sex

train['Sex'] = train['Sex'].map({'male':1,'female':2})

test['Sex'] = test['Sex'].map({'male':1,'female':2})
#查看head（）

train.head()
#抽取特征和目标值

features = train[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]

target = train[['Survived']]

test_features = test[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]

#划分训练集和交叉验证集

from sklearn.model_selection import train_test_split

x_train,x_cv,y_train,y_cv = train_test_split(features, target, test_size=0.25,random_state = 0)
#引入机器学习算法和accuracy函数

from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.linear_model import Perceptron

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis

#递归进行检验

algorithms = [GaussianNB(),

              LogisticRegression(),

              SVC(probability=True),

              Perceptron(),

              DecisionTreeClassifier(),

             RandomForestClassifier(),KNeighborsClassifier(),SGDClassifier(),GradientBoostingClassifier(),

             LinearDiscriminantAnalysis(),QuadraticDiscriminantAnalysis()]

name_alg = []

acc_alg = []

for clf in algorithms:

    name = clf.__class__.__name__

    name_alg.append(name)

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_cv)

    acc = round(accuracy_score(y_pred, y_cv) * 100, 2)

    acc_alg.append(acc)

#排序

result_alg = pd.DataFrame({'name of algorithm':name_alg,'accuracy of algorithm':acc_alg})

result_alg.sort_values(by='accuracy of algorithm', ascending = False)
#输出算法对test进行检测

best_alg = GradientBoostingClassifier()

best_alg.fit(x_train, y_train)

test_pred = best_alg.predict(test_features)

print(test_pred)