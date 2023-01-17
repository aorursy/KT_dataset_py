import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

sns.set()
data = pd.read_csv('/kaggle/input/gender-classification/Transformed Data Set - Sheet1.csv')
data.info()
data.shape
data.head()
_ = sns.countplot(x = data.Gender, y=None)
from sklearn.preprocessing import LabelEncoder



data1 = data.copy(deep = True)

le_color = LabelEncoder()

data1['Favorite Color'] = le_color.fit_transform(data['Favorite Color'])



le_genre = LabelEncoder()

data1['Favorite Music Genre'] = le_genre.fit_transform(data['Favorite Music Genre'])





le_beverage = LabelEncoder()

data1['Favorite Beverage'] = le_beverage.fit_transform(data['Favorite Beverage'])





le_drink = LabelEncoder()

data1['Favorite Soft Drink'] = le_drink.fit_transform(data['Favorite Soft Drink'])





le_gender = LabelEncoder()

data1['Gender'] = le_gender.fit_transform(data['Gender'])
data1.head()
X = data1.drop(['Gender'], axis = 1)

y = data1.Gender
from sklearn import tree



clf = tree.DecisionTreeClassifier(max_depth=4)

clf = clf.fit(X, y)
clf.score(X, y)
import matplotlib.pyplot as plt

plt.figure(figsize = (20,20))

_ = tree.plot_tree(clf.fit(X, y)) 
import graphviz 

dot_data = tree.export_graphviz(clf, out_file=None,feature_names=data1.drop(['Gender'],axis=1).columns, filled=True) 

graph = graphviz.Source(dot_data) 
graph
from sklearn import tree



clf = tree.DecisionTreeClassifier(max_depth=4, min_samples_split=6)

clf = clf.fit(X, y)
clf.score(X, y)
import graphviz 

dot_data = tree.export_graphviz(clf, out_file=None,feature_names=data1.drop(['Gender'],axis=1).columns, filled=True) 

graph = graphviz.Source(dot_data) 
graph
from sklearn import tree



clf = tree.DecisionTreeClassifier(max_depth=4, min_samples_split=6, min_samples_leaf=5)

clf = clf.fit(X, y)
dot_data = tree.export_graphviz(clf, out_file=None,feature_names=data1.drop(['Gender'],axis=1).columns, filled=True) 

graph = graphviz.Source(dot_data) 
graph