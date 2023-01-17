# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

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
train.info()
train.shape
train.isnull().sum()
test.isnull().sum()
test.shape
test.head()
train.describe()
train["Survived"].value_counts()
# sns.pairplot(train)
# plt.show()
# sns.heatmap(train)
890*.5

def bar_graph(features):
    survived = train[train["Survived"]==1][features].value_counts()
    dead = train[train["Survived"]==0][features].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index=['survived' , 'dead']
    df.plot(kind='bar',stacked=True)
bar_graph("Sex")
# (train[train["Survived"]==1]["Sex"].value_counts())
bar_graph("Pclass")
bar_graph("SibSp")
bar_graph("Parch")
bar_graph("Embarked")
train["Age"].fillna(train["Age"].mean() , inplace=True)
test["Age"].fillna(test["Age"].mean() , inplace=True)
# train["Age"] =train["Age"].fillna(train["Age"].mean() )
train.isnull().sum()
train.drop(["PassengerId","Name","Ticket","Cabin"], axis = 1 , inplace =True)
test.drop(["PassengerId","Name","Ticket","Cabin"], axis = 1 , inplace =True)
train.dropna(axis = 0,inplace=True)
test.dropna(axis = 0,inplace=True)
train
train.loc[train["Sex"]=="male" , "Sex"]=0
train.loc[train["Sex"]=="female" , "Sex"]=1

train.loc[train["Embarked"]=="S" , "Embarked"]=0
train.loc[train["Embarked"]=="Q" , "Embarked"]=1
train.loc[train["Embarked"]=="C" , "Embarked"]=2


test.loc[test["Sex"]=="male" , "Sex"]=0
test.loc[test["Sex"]=="female" , "Sex"]=1

test.loc[test["Embarked"]=="S" , "Embarked"]=0
test.loc[test["Embarked"]=="Q" , "Embarked"]=1
test.loc[test["Embarked"]=="C" , "Embarked"]=2
train.columns


train.head()
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
clf.fit(train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Embarked']], train["Survived"])
prediction = clf.predict(test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Embarked']])
clf.score(train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Embarked']], train["Survived"])

from sklearn.tree import DecisionTreeClassifier
sk  = DecisionTreeClassifier(criterion="entropy" , max_depth=5)
sk.fit(train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Embarked']], train["Survived"])
sk.predict(test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Embarked']])
sk.score(train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Embarked']], train["Survived"])
!pip install pydotplus
!pip install --upgrade scikit-learn==0.20.3
!pip install mglearn
import pydotplus

from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
dot_data = StringIO()
export_graphviz(sk,out_file=dot_data,filled=True,rounded=True)
['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Embarked']

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
!pip install --upgrade scikit-learn
from sklearn.ensemble import  RandomForestClassifier
rf = RandomForestClassifier(n_estimators=10 , criterion="entropy" , max_depth=5)
rf.fit(train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Embarked']], train["Survived"])
rf.score(train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Embarked']], train["Survived"])
