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
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error

from sklearn import preprocessing, svm, tree, linear_model, metrics

train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

train.head()
train.isnull().sum()
train.columns
train.describe()
train.dtypes
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

train["Sex"] = le.fit_transform(train["Sex"])

test["Sex"] = le.fit_transform(test["Sex"])





train
train["Parch"].unique()
train["Cabin"].unique()
plt.figure(figsize=(15,8))

ax = sns.kdeplot(train["Age"][train.Survived == 1], color="darkturquoise", shade=True)

sns.kdeplot(train["Age"][train.Survived == 0], color="lightcoral", shade=True)

plt.legend(['Survived', 'Died'])

plt.title('Density Plot of Age for Surviving Population and Deceased Population')

ax.set(xlabel='Age')

plt.xlim(-10,85)

plt.show()
sns.pairplot(train)
plt.figure(figsize=(40,8))

avg_survival_byage = train[["Age", "Survived"]].groupby(['Age'], as_index=False).mean()

g = sns.barplot(x='Age', y='Survived', data=avg_survival_byage, color="LightGreen")

plt.show()
plt.figure(figsize=(15,8))

ax = sns.kdeplot(train["Fare"][train.Survived == 1], color="darkturquoise", shade=True)

sns.kdeplot(train["Fare"][train.Survived == 0], color="lightcoral", shade=True)

plt.legend(['Survived', 'Died'])

plt.title('Density Plot of Fare for Surviving Population and Deceased Population')

ax.set(xlabel='Fare')

plt.xlim(-20,200)

plt.show()
sns.barplot('Embarked', 'Survived', data=train, color="teal")

plt.show()
sns.barplot('Sex', 'Survived', data=train, color="aquamarine")

plt.show()
corr = train.corr()

corr["Survived"].sort_values(ascending=False)
plt.figure(figsize = (20,20))

sns.heatmap(corr, annot=True)

plt.show()
train['Age'].fillna(train['Age'].median(skipna=True,),inplace=True)
train.isnull().sum()
train.drop(['Ticket'],axis=1, inplace=True)

train.drop('Cabin', axis=1, inplace=True)

train.drop('Embarked', axis=1, inplace=True)

train.drop('Name', axis=1, inplace=True)
train.dtypes
test.isnull().sum()
test.drop(['Ticket'],axis=1, inplace=True)

test.drop('Cabin', axis=1, inplace=True)

test.drop('Embarked', axis=1, inplace=True)

test.drop('Name',axis=1, inplace=True)
test.isnull().sum()
test["Age"].fillna(test["Age"].median(skipna=True,),inplace=True)

test["Fare"].fillna(test["Fare"].mean(skipna=True,),inplace=True)
test.isnull().sum()
test
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
Y_train = train["Survived"]

X_train = train.drop('Survived',axis=1,)

X_test = test
X_train, X_dev, Y_train, Y_dev = train_test_split(X_train, Y_train, test_size=0.15, random_state=1)

model.fit(X_train,Y_train)
Y_dev_pred = model.predict(X_dev)
print(mean_squared_error(Y_dev, Y_dev_pred))

print(r2_score(Y_dev, Y_dev_pred))
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)
print(metrics.accuracy_score(Y_dev, Y_dev_pred))

print(metrics.precision_score(Y_dev, Y_dev_pred))
Y_pred = model.predict(X_test)

Y_pred
my_submission = pd.DataFrame({'Id': test.PassengerId, 'Survived': Y_pred})

# you could use any filename. We choose submission here

my_submission.to_csv('gender_submission.csv', index=False)