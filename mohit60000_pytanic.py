# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import tree



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv', header=0)

test = pd.read_csv('../input/test.csv', header=0)
train["family_size"]=train["SibSp"]+train["Parch"]+1

train["Embarked"]=train["Embarked"].fillna("S")

train["Embarked"][train["Embarked"] == "S"] = 0

train["Embarked"][train["Embarked"] == "Q"] = 1

train["Embarked"][train["Embarked"] == "C"] = 2

train["Age"][train["Sex"]=='male']=train["Age"][train["Sex"]=='male'].fillna(train["Age"][train["Sex"]=='male'].median())

train["Age"][train["Sex"]=='female']=train["Age"][train["Sex"]=='female'].fillna(train["Age"][train["Sex"]=='female'].median())

train["Sex"][train["Sex"]=='male']=0 #Male=0 Female=1

train["Sex"][train["Sex"]=='female']=1

train["Age"][train["Age"]<=18]=0 #Age>18=1 Age<=18=0

train["Age"][train["Age"]>18]=1 #Age>18=1 Age<=18=0
getDeck = lambda x:np.nan if type(x)==float else x[0]

train['Deck']=train["Cabin"].map(getDeck)
mapping={'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'T':8, np.nan:0}

train['Deck']=train['Deck'].map(lambda x:mapping[x])
features=train[["Pclass","Sex","Age","Embarked","family_size","Deck"]].values

target=train[["Survived"]].values

max_depth=11

min_samples_split=2

mytree=tree.DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=1)

mytree=mytree.fit(features, target)
mytree.score(features, target)
test["family_size"]=test["SibSp"]+test["Parch"]+1

test["Embarked"]=test["Embarked"].fillna("S")

test["Embarked"][test["Embarked"] == "S"] = 0

test["Embarked"][test["Embarked"] == "Q"] = 1

test["Embarked"][test["Embarked"] == "C"] = 2

test["Age"][test["Sex"]=='male']=test["Age"][test["Sex"]=='male'].fillna(test["Age"][test["Sex"]=='male'].median())

test["Age"][test["Sex"]=='female']=test["Age"][test["Sex"]=='female'].fillna(test["Age"][test["Sex"]=='female'].median())

test["Sex"][test["Sex"]=='male']=0 #Male=0 Female=1

test["Sex"][test["Sex"]=='female']=1

test["Age"][test["Age"]<=18]=0 #Age>18=1 Age<=18=0

test["Age"][test["Age"]>18]=1 #Age>18=1 Age<=18=0

test['Deck']=test["Cabin"].map(getDeck)

test['Deck']=test['Deck'].map(lambda x:mapping[x])
test_features=test[["Pclass","Sex","Age","Embarked","family_size","Deck"]].values

my_prediction=mytree.predict(test_features)
PassengerId=np.array(test["PassengerId"]).astype(int)

my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])

my_solution.shape
my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])