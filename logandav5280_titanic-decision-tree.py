import pandas as pd
import numpy as np 

from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline 

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_val_score,GridSearchCV
import os
#giving you your current directory 

os.path.realpath('.')
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()
test.head()
train = train.drop(["Cabin","Name","PassengerId","Ticket"],1)
train.head()
len(train)
train.isnull().sum()
train = train.dropna()
train.isnull().sum()
train.head()
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()
train['Embarked'] = lb.fit_transform(train['Embarked'].astype(str))
train['Sex'] = lb.fit_transform(train['Sex'].astype(str))

model = DecisionTreeClassifier()
y = train[["Survived"]]
y.head()
x = train.drop(["Survived"],1)
x.head()
model.fit(x, y)
test['Embarked'] = lb.fit_transform(test['Embarked'].astype(str))
test['Sex'] = lb.fit_transform(test['Sex'].astype(str))

xtest = test.drop(["PassengerId","Cabin","Name","Ticket"],1)
xtest.isnull().sum()
xtest = xtest.fillna(xtest.mean())
xtest.isnull().sum()
out = model.predict(xtest)
test.PassengerId
test.head()
df =  pd.DataFrame(out,
test.PassengerId
)
df.to_csv('Titanic.csv')