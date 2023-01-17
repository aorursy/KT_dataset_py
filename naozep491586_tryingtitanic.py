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
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

import seaborn as sns

import matplotlib.pyplot as plt
train = pd.read_csv(r"/kaggle/input/titanic/train.csv")

test = pd.read_csv(r"/kaggle/input/titanic/test.csv")
train.head()
test.head()
print("train:",train.shape)

print("test:",test.shape)
train.info()

print("----------------------------------------")

print("----------------------------------------")

test.info()
train.describe()
test.describe()
plt.figure(figsize=(20,10))

sns.heatmap(train.isnull())
plt.figure(figsize=(20,10))

sns.heatmap(test.isnull())
train.isnull().sum()
test.isnull().sum()
fig = plt.figure()



ax1 = fig.add_subplot(1, 2, 1)

ax2 = fig.add_subplot(1, 2, 2)



# ヒストグラム

ax1.hist(train["Age"])

ax2.hist(test["Age"])



# 中央値と平均値を表示

ax1.axvline(train["Age"].mean(), color='k', linestyle='dashed', linewidth=1)

ax1.axvline(train["Age"].median(), color='r', linestyle='dashed', linewidth=1)



ax2.axvline(test["Age"].mean(), color='k', linestyle='dashed', linewidth=1)

ax2.axvline(test["Age"].median(), color='r', linestyle='dashed', linewidth=1)
train['Age']= train['Age'].fillna(train['Age'].median())

test['Age']= test['Age'].fillna(test['Age'].median())
print(train['Age'].isnull().sum())

print(test['Age'].isnull().sum())
train = train.drop("Cabin",axis=1)

test =  test.drop("Cabin",axis=1)
train = train.drop("Name",axis=1)

test = test.drop("Name",axis=1)

train = train.drop("Ticket",axis=1)

test = test.drop("Ticket",axis=1)
codes = {"male": 0, "female": 1}

train['Sex']= train['Sex'].map(codes)

test['Sex']= test['Sex'].map(codes)
codes = {"C": 0, "Q": 1,"S":2}

train['Embarked']= train['Embarked'].map(codes)

test['Embarked']= test['Embarked'].map(codes)
#todo とりあえず

train =train.fillna({"Fare":0})

test =test.fillna({"Fare":0})

train =train.fillna({"Embarked":0})

test =test.fillna({"Embarked":0})
train.describe()
sns.pairplot(train)
X_train = train.drop(['Survived','PassengerId'], axis=1)

Y_train = train["Survived"]

X_test  = test.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape


gbrt = GradientBoostingClassifier(random_state=0)

gbrt.fit(X_train, Y_train)



print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, Y_train)))
mlp = MLPClassifier(max_iter=1000, hidden_layer_sizes=[9,9,9],alpha=10, random_state=0)

mlp.fit(X_train_scaled, Y_train)



print("Accuracy on training set: {:.3f}".format(

    mlp.score(X_train_scaled, Y_train)))

forest = RandomForestClassifier(n_estimators=100, random_state=0)

forest.fit(X_train, Y_train)



print("Accuracy on training set: {:.3f}".format(forest.score(X_train, Y_train)))



y_pred = forest.predict(X_test)
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": y_pred

    })

submission.to_csv('titanic_1.csv', index=False)