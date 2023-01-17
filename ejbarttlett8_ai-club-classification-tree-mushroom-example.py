# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





import numpy as np # linear algebra

import pandas as pd # data orginization tool used for data processing/cleaning, CSV file I/O (e.g. pd.read_csv)

#Importining necceary sklearn libraries

from sklearn.preprocessing import LabelEncoder 

from sklearn.tree import DecisionTreeClassifier as skTree

from sklearn.model_selection import train_test_split 

from sklearn.tree import export_graphviz

#These Librares make it significantly easier to create graphs and other visual representations

import matplotlib.pyplot as plt

from mlxtend.plotting import plot_decision_regions

import graphviz



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



#This code just downloads the dataset from kaggle and stores it as a pandas array 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

mushroom = pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')

print(mushroom)

# Any results you write to the current directory are saved as output.
labelencoder = LabelEncoder()

for col in mushroom.columns:

    mushroom[col] = labelencoder.fit_transform(mushroom[col])

print(mushroom)
X = (mushroom.drop('class', axis = 1)).values

y = (mushroom['class']).values

print(X)

print("----")

print(y)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 1, stratify = y)

tree = skTree(criterion = 'entropy', max_depth = 5, random_state = 3)

model = tree.fit(X_train,y_train)
dot_data = export_graphviz(tree, out_file=None, feature_names = (mushroom.drop('class', axis = 1)).columns)

graph = graphviz.Source(dot_data) 

graph
pd.crosstab(mushroom['class'],mushroom['gill-color'])
predict = model.predict(X_test)



pd.crosstab(predict,y_test)
#percent BAD WRONG

print(str(12/(1251+40+12+1135)*100)+"% Poisonous Incorrectly Predicted as Edible")

predict = model.predict(X_train)



pd.crosstab(predict,y_train)
