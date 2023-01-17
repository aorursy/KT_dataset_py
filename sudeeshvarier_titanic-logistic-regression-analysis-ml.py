# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import cufflinks as cf
cf.go_offline()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/titanic_train.csv')
train.head()
#There is going to be some missing data in the dataset. So lets check that
train.isnull()
sns.heatmap(train.isnull(),yticklabels=False, cbar=False, cmap='viridis')
sns.set_style('whitegrid')
sns.countplot(x='Survived', data=train)
sns.countplot(x='Survived', hue='Sex', data=train, palette='RdBu_r')
sns.countplot(x='Survived', hue='Pclass', data=train)
sns.distplot(train['Age'].dropna(), kde=False, bins=30)
train['Age'].plot.hist(bins=20)
#doing the same thing using pandas native plotting. 
train.info()
sns.countplot(x='SibSp', data=train)
train['Fare'].hist(bins=40, figsize=(10,4))
plt.figure(figsize=(10,7))
sns.boxplot(x='Pclass',y='Age', data=train)
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age
    
#This function helps in getting the suitable possible age for the missing values in the Age column. 
train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis=1)
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
train.drop('Cabin', axis=1, inplace=True)
#Since there are loads of information missing in the cabin column, it is better we drop it completely
train.head()
plt.figure(figsize=(10,7))
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
train.dropna(inplace=True)
plt.figure(figsize=(10,7))
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
sex = pd.get_dummies(train['Sex'], drop_first=True)
embark = pd.get_dummies(train['Embarked'], drop_first=True)
train = pd.concat([train,sex,embark], axis=1)
train.head()
train.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)
train.head()
train.drop('PassengerId', axis=1, inplace=True)
train.head()
X = train.drop('Survived', axis=1)
y = train['Survived']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predictions)
train.head()
Pclass = pd.get_dummies(train['Pclass'], drop_first=True)
Pclass.head(20)
train = pd.concat([train,Pclass], axis=1)
train.head(20)
train.drop(['Pclass'], axis=1, inplace=True)
train.head(20)
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)
print(classification_report(y_test, predictions))

