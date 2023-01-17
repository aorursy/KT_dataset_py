# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
test = pd.read_csv("/kaggle/input/titanic/test.csv")
train = pd.read_csv("/kaggle/input/titanic/train.csv")
# Cмотрим на примеры и наименования столбцов
train.head()
# Смотрим итоговые данные по таблице
train.describe()
print("\ntrain null values\n")
total = train.isnull().sum().sort_values(ascending=False)
print(total)
print("\ntest null values\n")
total = test.isnull().sum().sort_values(ascending=False)
print(total)
# Убираем ненужные данные, имя нам не нужно, так как оно не влияет на выживаемость человека, номер билета также не несет в себе
# информации, а кабинки пассажиров имеют слишком мало значений
train = train.drop(['Name','Ticket','Cabin'], axis=1)
test = test.drop(['Name','Ticket','Cabin'], axis=1)
test.head()

train.info()
test.info()
train['Embarked'].describe()

train["Embarked"] = train["Embarked"].fillna("S")
train.info()
data = [train, test]
data = pd.concat(data, sort=False)
def add_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        return int(data[data["Pclass"] == Pclass]["Age"].mean())
    else:
        return Age

train['Age'] = train[['Age','Pclass']].apply(add_age,axis=1)
test['Age'] = test[['Age','Pclass']].apply(add_age,axis=1)
test['Age'] = test['Age'].astype(int)
train['Age'] = train['Age'].astype(int)
train.info()
# df1 = test[test.isna().any(axis=1)]
# df1
train.head()
data = [train, test]
data = pd.concat(data, sort=False)
fare_mean = data[data["Pclass"] == 3]["Fare"].mean()
test['Fare'] = test['Fare'].fillna(fare_mean)


test['Fare'] = test['Fare'].astype(int)
train['Fare'] = train['Fare'].astype(int)
train.info()
sns.countplot(x = 'Survived', data = train, hue = 'Sex')
sns.countplot(x = 'Survived', data = train, hue = 'Pclass')
sns.countplot(x = 'Survived', data = train)
sns.countplot(x = 'Survived', data = train, hue = 'SibSp')
train.head()
# sex = pd.get_dummies(train['Sex'],drop_first=True)
# embark = pd.get_dummies(train['Embarked'],drop_first=True)
city = {"S": 1, "C": 2, "Q": 3}
data = [train, test]
for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].map(city)
    
sex = {"male": 1, "female": 0}
for dataset in data:
    dataset["Sex"] = dataset["Sex"].map(sex)
# train = pd.concat([train, sex, embark], axis=1 )
# train.drop(['Sex', 'Embarked'], axis=1, inplace=True)
train.head(20)
# sex = pd.get_dummies(test['Sex'],drop_first=True)
# embark = pd.get_dummies(test['Embarked'],drop_first=True)

# test = pd.concat([test, sex, embark], axis=1 )
# test.drop(['Sex', 'Embarked'], axis=1, inplace=True)
test.head()
X_train = train.drop(["Survived","PassengerId"], axis=1)
Y_train = train["Survived"]
X_test  = test.drop("PassengerId", axis=1).copy()
X_train.shape, X_test.shape
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
print(acc_log)
from sklearn.model_selection import train_test_split

predictors = train.drop(['Survived', 'PassengerId'], axis=1)
target = train["Survived"]
x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.22, random_state = 0)
# Logistic Regression
from sklearn.linear_model import LogisticRegression
pid = test['PassengerId']
survived = logreg.predict(test.drop('PassengerId', axis=1))

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'PassengerId' : pid, 'Survived': survived })
output.to_csv('submission.csv', index=False)
