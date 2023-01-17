# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import missingno as msno

import warnings

warnings.filterwarnings('ignore')









# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

df_train = pd.read_csv("../input/train.csv")

sns.set_style("darkgrid")



# Any results you write to the current directory are saved as output.
df_train.head()
df_train.drop(['Cabin', 'Name','Ticket'], axis=1, inplace=True)
train = pd.get_dummies(df_train)
train['Age'].fillna(train['Age'].mean(),inplace=True)

n_train=train.astype(int)
plt.figure(figsize=(10,6))

plt.title('Survived Numbers')

sns.countplot(x ='Survived', data = df_train, hue='Survived', palette='inferno')
plt.figure(figsize=(10,6))

sns.barplot('Sex_female', 'Survived', data = n_train)

plt.title('Woman Survived (Orange)')



plt.figure(figsize=(10,6))

sns.barplot('Survived', 'Sex_male', data = n_train, palette='coolwarm')

plt.title('Man Survived (Orange)')



sns.lmplot('Fare', 'Survived', n_train, markers=False, hue='Pclass', size=8)

plt.title('Relation Survived and Fare')
plt.figure(figsize=(8,6))

plt.title("Dependents on board ",  fontsize=15)

plt.hist(n_train['Parch'], color='g',bins=6)

plt.xlabel("Numbers of 'Parch'")
df_train[df_train['Parch']==6]
plt.figure(figsize=(10,8))

plt.title('Pclass by Age', fontsize=14)

sns.swarmplot(x = 'Pclass', y = 'Age',data = n_train, hue='Pclass')

plt.figure(figsize=(10,8))

plt.title('Sex Female by Age', fontsize=14)

sns.boxplot(x = 'Sex_female', y = 'Age', data = n_train, hue='Survived',palette='coolwarm')

plt.figure(figsize=(10,8))



print('\n')

plt.title('Sex Male by Age', fontsize=14)

sns.boxplot(x = 'Sex_male', y = 'Age', hue='Survived', data = n_train,color='g')
plt.figure(figsize=(10,8))

plt.title('Relation Age and Fare', fontsize = 12)

sns.scatterplot(x = 'Age',y = 'Fare', data = n_train, palette='coolwarm', hue = 'Survived')
df_train[df_train['Age']==80]
plt.figure(figsize=(10,8))

plt.title('Relation Passenger and Fare', fontsize = 12)

sns.scatterplot(x = 'PassengerId', y = 'Fare', data = df_train, hue='Survived', palette='inferno')
df_train[df_train['Fare']>500]
df_train[df_train['Age']<1]
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
X = train.drop(['Survived'], axis=1)

y = train['Survived']
X_train, X_test, y_train,y_test = train_test_split(X,y, test_size=0.30)
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
predict = dt.predict(X_test)
from sklearn.metrics import classification_report

print (classification_report(y_test, predict))
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=500)
rfc.fit(X_train, y_train)
pred = rfc.predict(X_test)

print (classification_report(y_test, pred))