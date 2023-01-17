# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from mlxtend.plotting import plot_decision_regions

from sklearn.tree import DecisionTreeClassifier as skTree

from sklearn.model_selection import train_test_split 

from sklearn.tree import export_graphviz

import graphviz

import os

from sklearn import datasets
#iris2 = datasets.load_iris()

#pandas learn strat also .Catogorical



print(os.listdir("../input/iris"))

iris = pd.read_csv('../input/iris/Iris.csv')

print(iris.columns.values)
#Arranges data so that it can be inputed into sklearn using pandas

x = (iris[['PetalLengthCm', 'PetalWidthCm']]).values

y = (pd.Series(data = (iris["Species"]).values)).map(lambda i: 0 if i == "Iris-virginica" else (1 if i == "Iris-versicolor" else 2)).values



#Split the data into training and test blocks

#y is the outcome and x is the predictor

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 2, stratify = y)



#specify options for the tree (these can be adjusted to account for overfitting etc.) 

tree = skTree(criterion="gini", max_depth=3, random_state = 1)

#create the tree using the training data

model = tree.fit(x_train, y_train)
#below is plotting



#this combines training and test data

x_combined = np.vstack((x_train,x_test))

y_combined = np.hstack((y_train,y_test))

plot_decision_regions(x_combined, y_combined, clf=tree)



#these are labels for the plot

plt.xlabel('PetalWidthCm')

plt.ylabel('PetalLengthCm')

plt.legend(loc = 'upper left')

plt.show()
#ignore this stuff

#from pydotplus import graph_from_dot_data

#from sklearn.tree import export_graphviz

#'SepalLengthCm' 'SepalWidthCm' 'PetalLengthCm' 'PetalWidthCm'

#dont ignore the stuff below this comment tho



#print treeplot

dot_data = export_graphviz(tree, out_file=None)

graph = graphviz.Source(dot_data) 

graph





# Any results you write to the current directory are saved as output.
#this uses the 30% testing data to output predictions

predict = model.predict(x_test)



#this shows how many are correct or not correct

#It is not labeled, but there are three outcome variables

#100% accuracy (over-fitting) would have all the numbers on the diagonal (in the middle)

pd.crosstab(predict,y_test)



predict2 = model.predict(x_train)

pd.crosstab(predict2,y_train)