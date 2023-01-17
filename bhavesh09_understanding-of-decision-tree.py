import numpy as np

import pandas as pd

import xgboost as xgb

import matplotlib.pyplot as plt

%matplotlib inline
# This creates a pandas dataframe and assigns it to the train and test variables

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
# store target variable as Y

Y_train = train["Survived"]

train.drop(["Survived"], axis=1, inplace=True)
# Combine Train and test for ease

num_train = len(train)

all_data = pd.concat([train, test])
all_data.describe()
all_data.describe(include=['O'])
# Populating null fare value with median of train set

all_data["Fare"]=all_data["Fare"].fillna(train["Fare"].median())
# Populating null age value with median of train set

all_data["Age"]=all_data["Age"].fillna(train["Age"].median())
# Populating missing embarked with most frequent value - S

all_data["Embarked"]=all_data["Embarked"].fillna("S")
# Drop cabin due to too many null values

all_data.drop(["Cabin"], axis=1, inplace=True)
from sklearn import preprocessing 

#convert objects / non-numeric data types into numeric

for f in all_data.columns:

    if all_data[f].dtype=='object':

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(all_data[f].values)) 

        all_data[f] = lbl.transform(list(all_data[f].values))
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

from sklearn import tree
# Prepare train and test for prediction

X_train = all_data[:num_train]

X_test = all_data[num_train:]
# create validation set

X_train, X_cv, y_train, y_cv = train_test_split( X_train, Y_train, test_size = 0.3, random_state = 100)
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,

                               max_depth=3, min_samples_leaf=5)

clf_gini.fit(X_train, y_train)
# Graphviz is used to build decision trees

from sklearn.tree import export_graphviz

from sklearn import tree
# This statement builds a dot file.

cols = list(X_train.columns.values)

tree.export_graphviz(clf_gini, out_file='tree.dot',feature_names  = cols)  
y_pred = clf_gini.predict(X_cv)
def score_in_percent (a,b):

    return (sum(a==b)*100)/len(a)
score_in_percent(y_pred,y_cv)
# Let's do little tweak, and drop name and Cabin features

X_train.drop(["Name"], axis=1, inplace=True)

X_cv.drop(["Name"], axis=1, inplace=True)

X_test.drop(["Name"], axis=1, inplace=True)
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,

                               max_depth=3, min_samples_leaf=5)

clf_gini.fit(X_train, y_train)

y_pred = clf_gini.predict(X_cv)

score_in_percent(y_pred,y_cv)
y_test_pred = clf_gini.predict(X_test)
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": y_test_pred

    })

submission.to_csv('submission_with_3depth.csv', index=False) # LB : 0.74163
# This statement builds a dot file.

cols = list(X_train.columns.values)

tree.export_graphviz(clf_gini, out_file='tree_won.dot',feature_names  = cols)  
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,

                               max_depth=4, min_samples_leaf=5)

clf_gini.fit(X_train, y_train)

y_pred = clf_gini.predict(X_cv)

score_in_percent(y_pred,y_cv)
# This statement builds a dot file.

cols = list(train.columns.values)

tree.export_graphviz(clf_gini, out_file='tree_four.dot',feature_names  = cols)  
y_test_pred = clf_gini.predict(X_test)

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": y_test_pred

    })

submission.to_csv('submission_with_4depth.csv', index=False) # LB : 0.73206