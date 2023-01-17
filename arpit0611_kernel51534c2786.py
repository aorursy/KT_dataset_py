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
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
train.isnull().sum(axis=0)
train["Age"].median(axis=0,skipna=True)
train["Age"]= train["Age"].fillna(train["Age"].median())
train.isnull().sum(axis=0)
test["Age"]= test["Age"].fillna(test["Age"].median())
test.info()
"""
    import pandas as pd

    from sklearn import tree
     
    train = pd.read_csv("C:/Users/CHANDRIMA/Desktop/topics/titanic/train.csv")
    clean_data(train)
     
    target = train["Survived"].values
    features = train[["Pclass", "Age", "Fare", "Embarked", "Sex", "SibSp", "Parch"]].values
     
    decision_tree = tree.DecisionTreeClassifier(random_state = 1)
    decision_tree_ = decision_tree.fit(features, target)
     
    print(decision_tree_.score(features, target)) 
    """
from sklearn import linear_model
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier


y = train["Survived"]

features = ["Age","Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train[features])
X_test = pd.get_dummies(test[features])

model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
