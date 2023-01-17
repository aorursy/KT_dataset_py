# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # 
from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn.datasets import load_iris
from graphviz import Source
from IPython.display import SVG

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
df = pd.read_csv("../input/diabetes.csv")

# drop NaN values
df = df.dropna(thresh = 1)

# correlation matrix
url = "../input/diabetes.csv"
data = pd.read_csv(url)
correlations = data.corr()
# plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
plt.show()

# drop skin thickness column
df = df.drop('SkinThickness', 1)

# Not changing true and false values as they already fit the required format

# showing ratio of true and false data
true = (df['Outcome'] == 1).sum()
false = (df['Outcome'] == 0).sum()
truePercentage = true / (true + false)
falsePercentage = false / (true+false)
print("Ratio of true to false outcomes: ")
print("True: ")
print(truePercentage)
print("False: ")
print(falsePercentage)
df

# set two arrays for the attributes and outcomes
X = df.values[:, 0:6]
Y = df.values[:, 7]
# splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.7, random_state = 100)



# Any results you write to the current directory are saved as output.
# create a new model for the ID3 decition tree training
id3_model = tree.DecisionTreeClassifier(criterion = "entropy")
# Train the ID3 decision tree
id3_model.fit(X_train, y_train)


# Graph the ID3 tree
iris = load_iris()
graph = Source( tree.export_graphviz(id3_model, out_file = None))
SVG(graph.pipe(format='svg'))
# Create a new model using the Default CART algorithm to emulate C4.5
C45_model = tree.DecisionTreeClassifier(criterion = "gini")
# Train the ID3 decision tree
C45_model.fit(X_train, y_train)
# Graph the C45 tree
graph2 = Source( tree.export_graphviz(C45_model, out_file = None))
SVG(graph2.pipe(format='svg'))