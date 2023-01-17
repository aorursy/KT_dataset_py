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
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

gender_submission = pd.read_csv('../input/titanic/gender_submission.csv')

#train.head()

#test.head()

#gender_submission.head()
#import pandas_profiling

#train.profile_report()
import matplotlib.pyplot as plt

#Ageと目的変数(Survived)の関係

#plt.hist(train.loc[train['Survived'] == 0, 'Age'].dropna(),bins=30,alpha=0.5,label = '0')

#plt.hist(train.loc[train['Survived'] == 1, 'Age'].dropna(),bins=30,alpha=0.5,label = '1')

#plt.xlabel('Age')

#plt.ylabel('count')

#plt.legend(title = 'Survived')
import seaborn as sns

#SibSpと目的変数の関係

#sns.countplot(x = 'SibSp', hue = 'Survived', data = train)

#plt.legend(loc = 'upper right', title = 'Survived')
#Parchと目的変数との関係

#sns.countplot(x = 'Parch', hue = 'Survived', data = train)

#plt.legend(loc = 'upper right', title = 'Survived')
#Fareと目的変数との関係

#plt.hist(train.loc[train['Survived'] == 0, 'Fare'].dropna(), range = (0,250), bins = 25, alpha = 0.5, label = '0')

#plt.hist(train.loc[train['Survived'] == 1, 'Fare'].dropna(), range = (0,250), bins = 25, alpha = 0.5, label = '1')

#plt.xlabel('Fare')

#plt.ylabel('count')

#plt.legend(title = 'Survived')

#plt.xlim(-5,250)
#Pclassと目的変数との関係

#sns.countplot(x = 'Pclass', hue = 'Survived',data = train)
#Sexと目的変数の関係

#sns.countplot(x = 'Sex', hue = 'Survived',data = train)
#Embarkedと目的変数の関係

#sns.countplot(x = 'Embarked', hue = 'Survived',data = train)
data = pd.concat([train,test], sort = False) #trainとtestを縦に結合



data['Sex'].replace(['male','female'],[0,1], inplace = True) #maleとfemaleを0と1に変換



data['Fare'].fillna(np.mean(data['Fare']),inplace = True) #Fareの欠損値を平均で補完



data['Age'].fillna(data['Age'].median(), inplace = True) #Ageの欠損値を中央値で補完

'''

age_avg = data['Age'].mean()

age_std = data['Age'].std()



data['Age'].fillna(np.random.randint(age_avg - age_std, age_avg + age_std), inplace=True) #Ageの欠損値をを平均年齢±標準偏差の乱数で補完

'''



data['Embarked'].fillna(('S'), inplace=True) #Embarkedの欠損値を最頻値Sで補完

data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int) #アルファベットを0,1,2(int)に変換



delete_columns = ['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin']

data.drop(delete_columns, axis=1, inplace=True) #最初の項目の列の削除
train = data[:len(train)]

test = data[len(train):] #データの再分割

print(train.shape)

print(test.shape)
y_train = train['Survived']

X_train = train.drop('Survived', axis=1)

X_test = test.drop('Survived', axis=1)

print(y_train.shape)

print(X_train.shape)

print(X_test.shape)
#from sklearn.linear_model import LogisticRegression as LR



#clf = LR(penalty = 'l2', solver = 'sag', random_state = 0)

#clf.fit(X_train, y_train)
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators = 100, max_depth = 2, random_state = 0)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(y_pred.shape)
sub = pd.read_csv('../input/titanic/gender_submission.csv')

sub['Survived'] = list(map(int, y_pred))

sub.to_csv('submission.csv', index = False)