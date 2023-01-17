import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn import model_selection



import seaborn as sns



%matplotlib inline
df = pd.read_csv('../input/HR_comma_sep.csv')
# Printing the list of columns in the dataset



df.columns
df.head()
df.describe()
# The data of the users who have left

df[df['left'] == 1].describe()
# The data of the users who have been retained

df[df['left'] == 0].describe()
df['sales'].describe()

df['sales'][df['left'] == 1].value_counts(normalize = True)
df['sales'][df['left'] == 0].value_counts(normalize = True)
df['salary'][df['left'] == 0].value_counts(normalize = True)
df['salary'][df['left'] == 1].value_counts(normalize = True)
df['salary'] = df['salary'].map({'low':0,'medium':1,'high':2})
df.columns
train = df.drop([ 'left', 'sales'], 1) # number_project, #last_evaluation



test = df['left']
X_train, X_test, y_train, y_test= model_selection.train_test_split(train, test, test_size=0.3)
# Decision trees Classifier



clf = DecisionTreeClassifier(random_state=0)

clf.fit(X_train, y_train)



accuracy = clf.score(X_test,y_test)

print(accuracy)
clf2 = RandomForestClassifier(random_state = 0)

clf2.fit(X_train, y_train)



accuracy2 = clf2.score(X_test,y_test)

print(accuracy2)