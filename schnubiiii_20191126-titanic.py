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



# what I imported additionally: 

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.ticker import PercentFormatter

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

%matplotlib inline
submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
train.head()
train.info()
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

plt.show()
train.drop(['Cabin', 'Name', 'Ticket'],axis=1,inplace=True)

train.head()
test.head()
test.drop(['Cabin', 'Name', 'Ticket'],axis=1,inplace=True)
sns.countplot(x = 'Survived', hue = 'Sex', data = train)

plt.title("Survival Rate by male or female passengers")

plt.show()
ax=sns.barplot(x='Sex',y = 'Survived', hue= 'Pclass',data=train,palette='rainbow')

ax.yaxis.set_major_formatter(PercentFormatter(1.0)) #formating the y-achsis to percentage 

plt.title("Survival Rate by Gender and Boarding Class")

plt.show()
ax = sns.barplot(x='Pclass', y='Survived', color = 'blue', data = train )

ax.yaxis.set_major_formatter(PercentFormatter(1.0))

plt.title("Survival Rate by Boarding Class")

plt.show()
age_mean = train.Age.mean()

age_mean

train.fillna({'Age': age_mean}, inplace=True)

train.head()
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

plt.show()
age_mean2 = test.Age.mean()

age_mean2

test.fillna({'Age': age_mean2}, inplace=True)

test.head()
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')

plt.show()
sex = pd.get_dummies(train['Sex'])

embark = pd.get_dummies(train['Embarked'])
train = pd.concat([train,sex,embark],axis=1)

train.drop(['Sex', 'Embarked'], axis=1,inplace=True)

train.head()
X = train.drop(['Survived'], axis = 1)

y = train['Survived']



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30, random_state=101)
Log = LogisticRegression()

Log.fit(X_train, y_train)

pred= Log.predict(X_test)

print(Log.score(X_test, y_test))
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

scaler.fit(X_train)



X_train1 = scaler.transform(X_train)

X_test1 = scaler.transform(X_test)
from sklearn.svm import SVC



SVM = SVC(kernel = "rbf")

SVM.fit(X_train1, y_train)



print(SVM.score(X_test1, y_test))
from sklearn.neighbors import KNeighborsClassifier



model = KNeighborsClassifier(n_neighbors = 3, )

model.fit(X_train1, y_train)



print(model.score(X_test1, y_test))
from sklearn.ensemble import RandomForestClassifier



RFF = RandomForestClassifier(criterion = "entropy", n_estimators = 100)

RFF.fit(X_train, y_train)



print(model.score(X_test, y_test))
sex = pd.get_dummies(test['Sex'])

embark = pd.get_dummies(test['Embarked'])

test = pd.concat([test,sex,embark],axis=1)

test.drop(['Sex', 'Embarked'], axis=1,inplace=True)

test.head()
#df[df.isna().any(axis=1)]

test[test.isna().any(axis=1)]
age_fare2 = test.Fare.mean()

age_fare2

test.fillna({'Fare': age_fare2}, inplace=True)

test[test.isna().any(axis=1)] # checking if the filling of the value worked
pred2= SVM.predict(test)
submission['PassengerId'] = test['PassengerId']

submission['Survived'] = pred2

submission.head()
submission.to_csv("titanic_submissionSVM.csv", index=False)