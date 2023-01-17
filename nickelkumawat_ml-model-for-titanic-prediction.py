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
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
pd.pandas.set_option('display.max_columns',None)
train=pd.read_csv("/kaggle/input/titanic/train.csv")
test=pd.read_csv("/kaggle/input/titanic/test.csv")
train.head()
train.shape
test.shape
train.info()
#number of passengers survived
sum(list(train.Survived))
train.describe()
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Age', bins=20)
plt.figure(figsize=(14, 6))
plt.subplot(1,2,1)
cols = ['blue', 'lightcoral']
train['Sex'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',shadow=True, colors=cols)
plt.title('Total Male/Female onboard')
plt.subplot(1,2,2)
sns.barplot(x="Sex", y="Survived", data=train,palette='plasma')
plt.title('Sex vs Survived')
plt.ylabel("Survival Rate")
plt.show()

train.groupby(["Survived"]).Fare.mean()

#Correlation:  Its the most basic way to find relation between any two quantities.
corr = train.corr()

plt.figure(figsize=(15,15))

sns.heatmap(corr,vmax=0.9,square=True)
plt.show()
train.corr()["Survived"]
train.isnull().mean()
#lets drop cabin
train.drop("Cabin", axis = 1, inplace = True)
test.drop("Cabin", axis = 1, inplace = True)
#age can be filled by median
train["Age"].fillna(train["Age"].median(), inplace = True)
test["Age"].fillna(test["Age"].median(), inplace = True) 
#embarked can be filled by median/or mode
train['Embarked'].value_counts(normalize=True)
Embarked_mode=train['Embarked'].mode()[0]
Embarked_mode

train["Embarked"].fillna("S", inplace = True)
test.isnull().mean()
#fare can be filled with median
test["Fare"].fillna(test["Fare"].median(), inplace = True)
test.isnull().mean()
train.isnull().mean()
train['Sex'] = train['Sex'].map({'female': 0, 'male': 1})
test['Sex']= test['Sex'].map({'female': 0, 'male': 1})

train['Embarked'] = train['Embarked'].map({'S': 0, 'C': 1,'Q': 2})
test['Embarked']= test['Embarked'].map({'S': 0, 'C': 1,'Q': 2})
train.drop(["Name","Ticket"], axis = 1, inplace = True)
test.drop(["Name","Ticket"], axis = 1, inplace = True)
train["Family"] = train["SibSp"] + train["Parch"] + 1
test["Family"] = test["SibSp"] + test["Parch"] + 1
train=train.drop(["SibSp","Parch"],axis=1)
test=test.drop(["SibSp","Parch"],axis=1)
print(train.shape)
print(test.shape)
scaler = StandardScaler()

train[['Age','Fare']] = scaler.fit_transform(train[['Age','Fare']])
test[['Age','Fare']] = scaler.transform(test[['Age','Fare']])

train.head()
test.head()

X_train = train.drop(['Survived','PassengerId'], axis=1)
y_train = train["Survived"]
X_test  = test.drop("PassengerId", axis=1)
X_train.shape, y_train.shape, X_test.shape
# Logistic Regression

LR = LogisticRegression()
LR.fit(X_train, y_train)

# Making Predictions
y_pred = LR.predict(X_test)
# Calculating the Accuracy of the model.

print("Accuracy:",round(LR.score(X_train, y_train)*100,2))
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
Y_pred = random_forest.predict(X_test)
#Checking accuracy
random_forest.score(X_train, y_train)

acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
acc_random_forest
Features=['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Family']
feature_importance = pd.Series(random_forest.feature_importances_,index=Features).sort_values(ascending=False)
feature_importance
combine = [train, test]


train[['Family', 'Survived']].groupby(['Family'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['Family'] == 1, 'IsAlone'] = 1

train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
X_train = train.drop(['Survived','PassengerId'], axis=1)
y_train = train["Survived"]
X_test  = test.drop("PassengerId", axis=1)
X_train.shape, y_train.shape, X_test.shape
random_forest_2 = RandomForestClassifier(n_estimators=100)
random_forest_2.fit(X_train, y_train)
Y_pred = random_forest_2.predict(X_test)
#Checking accuracy
random_forest_2.score(X_train, y_train)

acc_random_forest_2 = round(random_forest_2.score(X_train, y_train) * 100, 2)
acc_random_forest_2
Features=['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Family',"IsAlone"]
feature_importance = pd.Series(random_forest_2.feature_importances_,index=Features).sort_values(ascending=False)
feature_importance
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': Y_pred})

output.to_csv('my_submission_2.csv', index=False)
print("Your submission was successfully saved!")
# Instantiate our model
xg = XGBClassifier()
xg.fit(X_train, y_train)
xg_predictions = xg.predict(X_test)
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': xg_predictions})

output.to_csv('my_submission_3.csv', index=False)
print("Your submission was successfully saved!")

xg.score(X_train,y_train)
