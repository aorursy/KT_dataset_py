# Importing libraries

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn import tree

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score


# Importing dataset

emp_data = pd.read_csv('../input/employee-attrition/WA_Fn-UseC_-HR-Employee-Attrition.csv')

emp_data.head()
emp_data.shape
# Splitting the dataset to get dependent and independent variables

X = emp_data.iloc [:, [4,6,7,10,13,14,15,16,22,24,25,30]]

Y = emp_data.iloc [:, 1]

X.head()

# Splitting into train and test sets

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Encoding categorial variables

one_hot_encoded_training_predictors = pd.get_dummies(X_train)

one_hot_encoded_test_predictors = pd.get_dummies(X_test)

X_train, X_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors, join='left', axis=1)

X_train.head()
X_train.shape
# Training with gini

clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 0, min_samples_leaf = 5)

clf_gini = clf_gini.fit(X_train, Y_train)

# Prediction

Y_pred = clf_gini.predict (X_test)

print ("Accuracy: ", accuracy_score(Y_test, Y_pred)*100)
# Training with entropy

clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 0, min_samples_leaf = 5)

clf_entropy = clf_entropy.fit(X_train, Y_train)

# Prediction

Y_pred = clf_entropy.predict (X_test)

print ("Accuracy: ", accuracy_score(Y_test, Y_pred)*100)