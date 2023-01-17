# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set()



# plotly library

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)





import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=FutureWarning)

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")
train.head()
test.head()
train.isnull().sum()
test.isnull().sum()
plt.figure(figsize=(10,6))

sns.heatmap(train.isnull())

plt.show()
bins = np.arange(0,80,5)

g = sns.FacetGrid(train,row="Sex",col="Pclass",hue="Survived")

g.map(sns.distplot,"Age",bins=bins,kde=False)

g.add_legend()

plt.show()
sns.barplot(x="Pclass",y="Survived",data=train)

plt.title("Survival")

plt.xlabel("Pclass")

plt.ylabel("Survival Rate")

plt.show()
sns.barplot(x="Sex",y="Survived",hue="Pclass",data=train)

plt.ylabel("Survival")

plt.show()
sns.barplot(x='Embarked', y='Survived', data=train)

plt.ylabel("Survival Rate")

plt.title("Survival as function of Embarked Port")

plt.show()
train.Age.median()
test.Age.median()
train["TravelAlone"] = train["SibSp"] + train["Parch"]
test["TravelAlone"] = test["SibSp"] + test["Parch"]
train.head()
list = ["Name","SibSp","Parch","Ticket","Fare","Cabin"]

train.drop(list,axis=1,inplace=True)
train.head()
list = ["Name","SibSp","Parch","Ticket","Fare","Cabin"]

test.drop(list,axis=1,inplace=True)
test.head()
train.drop("PassengerId",axis=1,inplace=True)

test.drop("PassengerId",axis=1,inplace=True)
train.head()
train.isnull().sum()
test.isnull().sum()
train["Age"].fillna(train["Age"].median(),inplace=True)

train["Embarked"].fillna(train["Embarked"].mode(),inplace=True)

test["Age"].fillna(test["Age"].median(),inplace=True)
plt.figure(figsize=(10,6))

sns.heatmap(train.isnull());
plt.figure(figsize=(10,6))

sns.heatmap(test.isnull());
sns.countplot(train.Embarked)
sns.countplot(train.Sex)
sns.countplot(train.TravelAlone)
sns.kdeplot(train.Age);
sns.kdeplot(train.TravelAlone);
plt.figure(figsize=(16,12))

sns.pairplot(train,hue="Survived")
sns.barplot(x="Sex",y="Survived",data=train);
train.info()
train = pd.get_dummies(train, drop_first=True)
test = pd.get_dummies(test, drop_first=True)
train.head()
corr = train.corr()
plt.figure(figsize= (14,12))

sns.heatmap(corr,square=True,annot=True,cmap="Blues")

plt.show()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
train.head()
X = train[['Pclass', 'Age', 'TravelAlone', 'Sex_male', 'Embarked_Q',

       'Embarked_S']]

y = train["Survived"]
X.head()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)
X_train.head()
logreg = LogisticRegression()

logreg.fit(X_train,y_train)

y_pred = logreg.predict(X_test)
print(accuracy_score(y_test,y_pred))
from sklearn.tree import DecisionTreeClassifier

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

tree = DecisionTreeClassifier()

tree.fit(X_train,y_train)

y_pred = tree.predict(X_test)
print(accuracy_score(y_test,y_pred))