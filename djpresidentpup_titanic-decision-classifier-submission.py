#Libraries
import pandas as pd
import numpy as np

#Visualizing
from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline

#Machine learning libraries
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, GridSearchCV
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()
test.head()
#PREPROCESSING
train = train.fillna(train.median())
lb = LabelEncoder()
train['Embarked'] = lb.fit_transform(train['Embarked'].astype(str))
train['Sex'] = lb.fit_transform(train['Sex'].astype(str))

dt = DecisionTreeClassifier()
x = train.drop(['Survived', 'PassengerId', 'Cabin', 'Ticket','Name'], axis = 1)
y = train['Survived']
x.head()
#Use model 
tree_model = dt.fit(x, y)
#Cross validates
cross_val_score(tree_model, x, y, cv = 30).mean()
sns.set(color_codes=True)
#Histogram of ages on the ship 
sns.distplot(train["Age"], kde = False, rug = True)