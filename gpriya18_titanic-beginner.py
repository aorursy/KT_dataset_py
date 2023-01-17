# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('/kaggle/input/titanic/train.csv')

test=pd.read_csv('/kaggle/input/titanic/test.csv')

train.head()



train.Survived.value_counts(normalize=True).plot(kind="bar",alpha=0.5)

plt.show()
train.Survived.value_counts()
plt.scatter(train.Survived,train.Age,alpha=0.1)

plt.show()
train.Pclass.value_counts(normalize=True).plot(kind="bar",alpha=0.5)

plt.show()
for x in [1,2,3]:

    train.Age[train.Pclass==x].plot(kind="kde")

plt.show()
train.Survived[train.Sex=="male"].value_counts(normalize=True).plot(kind="bar",alpha=0.5)

plt.title("men survived or ded")
train.Survived[train.Sex=="female"].value_counts(normalize=True).plot(kind="bar",alpha=0.5)

plt.title("women survived or ded")
train.Sex[train.Survived==1].value_counts(normalize=True).plot(kind="bar",alpha=0.5)

plt.title("sex survived")
for x in [1,2,3]:

    train.Survived[train.Pclass==x].plot(kind="kde")

plt.legend(("1","2","3"))

plt.show()
train.Survived[(train.Sex=="male") & (train.Pclass==3)].value_counts(normalize=True).plot(kind="bar",alpha=0.5)

plt.title("poor men")

plt.show()
train.Survived[(train.Sex=="male") & (train.Pclass==1)].value_counts(normalize=True).plot(kind="bar",alpha=0.5)

plt.title("rich men")

plt.show()
train.Survived[(train.Sex=="female") & (train.Pclass==3)].value_counts(normalize=True).plot(kind="bar",alpha=0.5)

plt.title("poor women")

plt.show()
train.Survived[(train.Sex=="female") & (train.Pclass==1)].value_counts(normalize=True).plot(kind="bar",alpha=0.5)

plt.title("rich women")

plt.show()
train.Survived[train.Sex=="male"].value_counts()
train.loc[(train.Sex=='female') & (train.SibSp==1)]
train.head()
type(train['Survived'])

#g=sns.pairplot(train)

#data cleaning

train["Fare"]= train["Fare"].fillna(train["Fare"].dropna().median())

train["Age"]= train["Age"].fillna(train["Age"].dropna().median())



train.loc[train["Sex"]=="female","Sex"]=1

train.loc[train["Sex"]=="male","Sex"]=0



train["Embarked"]= train["Embarked"].fillna('S')

train.loc[train["Embarked"]=="S", "Embarked"]=0

train.loc[train["Embarked"]=="C", "Embarked"]=1

train.loc[train["Embarked"]=="Q", "Embarked"]=2

train.head()







#data cleaning test

test["Fare"]= test["Fare"].fillna(test["Fare"].dropna().median())

test["Age"]= test["Age"].fillna(test["Age"].dropna().median())



test.loc[test["Sex"]=="female","Sex"]=1

test.loc[test["Sex"]=="male","Sex"]=0



test["Embarked"]= test["Embarked"].fillna('S')

test.loc[test["Embarked"]=="S", "Embarked"]=0

test.loc[test["Embarked"]=="C", "Embarked"]=1

test.loc[test["Embarked"]=="Q", "Embarked"]=2

test.head()





from sklearn.linear_model import LogisticRegression 

import utils



#utils.clean_data(train)

target=train["Survived"]

features=train[["Pclass","Age","Fare","Embarked","Sex","SibSp","Parch"]]



classifier = LogisticRegression(random_state = 0) 

classifier.fit(features, target) 

print(classifier.score(features,target))
from sklearn.linear_model import LogisticRegression 

import utils

from sklearn import tree,model_selection



#utils.clean_data(train)

target=train["Survived"]

feature=["Pclass","Age","Fare","Embarked","Sex","SibSp","Parch"]

features=train[feature]



classifier = tree.DecisionTreeClassifier(random_state = 1) 

classifier=classifier.fit(features, target) 

print(classifier.score(features,target))



scores=model_selection.cross_val_score(classifier,features,target,scoring='accuracy',cv=50)

print(scores)

print(scores.mean())


#utils.clean_data(train)

target=train["Survived"]

feature=["Pclass","Age","Fare","Embarked","Sex","SibSp","Parch"]

features=train[feature]



classifier = tree.DecisionTreeClassifier(random_state = 1,max_depth=7,min_samples_split=2) 

classifier=classifier.fit(features, target) 

print(classifier.score(features,target))



scores=model_selection.cross_val_score(classifier,features,target,scoring='accuracy',cv=50)

print(scores)

print(scores.mean())
import graphviz 

dot_data = tree.export_graphviz(classifier, out_file=None,feature_names=feature)  

graph = graphviz.Source(dot_data)  

graph 
classifier.predict(test[feature])
train.head()
from sklearn.ensemble import RandomForestClassifier



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(features, target)

Y_pred = random_forest.predict(test[feature])

random_forest.score(features, target)
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": Y_pred

    })





submission

#submission.to_csv('../output/submission.csv', index=False)
import os

os.chdir(r'/kaggle/working')

submission.to_csv(r'submission.csv', index=False)

from IPython.display import FileLink

FileLink(r'submission.csv')