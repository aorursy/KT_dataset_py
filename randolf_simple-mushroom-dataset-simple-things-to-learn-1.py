# Notebook for beginners. Easy ML task as Iris dataset.

# Used in this notebook: Basic dataframe methods to select data; basic scikit-learn models:

# logistic regression, gradient boosting, decision tree and Knn;

# some visualization: matplotlib and graphviz.

# There are no  NaN (missing) values. 

# Binary Classification.
import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import pandas as pd

from IPython.display import display # display() tool from IPython for dataframe visualization

#plt.rc('font', family='Verdana') # you can comment this

from sklearn import preprocessing

# basic import Cell. 
data = pd.read_csv("../input/mushrooms.csv")
display(data.head()) #Show us 5 first rows

# or just data.head()

#As expected (read description), features are indicated by letters.
data_encoded = preprocessing.LabelEncoder() #So, lets code them into numeric categories. Used scikit 

for column in data.columns[1:]: #OneHotEncoder. Another way - pandas.get_dummies. Code 

    data[column] = data_encoded.fit(data[column]).transform(data[column])

# data is our pd.Dataframe object and  .columns[1:] are our features

# 1st is column number 0 (class) and it contains our labels

# I "separate" labels and features only for example in this case

data['class'] = data_encoded.fit(data['class']).transform(data['class']) # Encode 1st column with labels

#... .fit(...)

#... .transfrom(..)

# 2 rows will give same result.

display(data.head()) #Seems good
features = data.iloc[:,1:] #.loc and .iloc for data selecting. Choose all rows and columns from position 2

X = features.values #  Extract features values

y = data.iloc[:,0] # Extract labels (column number 0) values

print("Array form (X): {} Array form (y): {}".format(X.shape, y.shape)) # Check for mistakes 

#22 features and 1 class. 8124 rows

# Now split data with the simplest method (train_test_split):

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y) 

# Cells below are representing some models that can be used in binary classification
#Playground time!

# Logistic Regression

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(random_state=0) 

logreg.fit(X_train, y_train)



print("Accuracy, training set: {:.3f}".format(logreg.score(X_train, y_train)))

print("Accuracy, test set: {:.3f}".format(logreg.score(X_test, y_test)))
#Gradient boosting

from sklearn.ensemble import GradientBoostingClassifier

gbrt = GradientBoostingClassifier(random_state=0)

gbrt.fit(X_train, y_train)

print("Accuracy, training set: {:.3f}".format(gbrt.score(X_train, y_train)))

print("Accuracy, test set: {:.3f}".format(gbrt.score(X_test, y_test)))

#Also we can visualizate feature importances (matplotlib help us)

def plot_feature_importances(model):

    n_features = features.columns.size

    plt.barh(range(n_features), model.feature_importances_, align='center')

    plt.yticks(np.arange(n_features), features.columns)

    plt.xlabel("Feature importance")

    plt.ylabel("Feature")

plot_feature_importances(gbrt)
# Decision tree

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(random_state=0) #try max_depth = 1..8, from rough to 

# (very) sensitive model

tree.fit(X_train, y_train)

print("Accuracy, training set: {:.3f}".format(tree.score(X_train, y_train)))

print("Accuracy, test set: {:.3f}".format(tree.score(X_test, y_test)))

# And visualizate how decision tree works
from sklearn.tree import export_graphviz

export_graphviz(tree, out_file="tree.dot", class_names = ["edible", "poisonous"], #edible = 0,               

feature_names = features.columns, impurity=False, filled=True) # poisonous = 1, right? 

import graphviz

with open("tree.dot") as file:

    dot_graph = file.read()

graphviz.Source(dot_graph)
# Knn

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=7) #n_neighbors=1..scary number

knn.fit(X_train, y_train)

print("Accuracy, training set: {:.3f}".format(knn.score(X_train, y_train)))

print("Accuracy, test set: {:.3f}".format(knn.score(X_test, y_test)))