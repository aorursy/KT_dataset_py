# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

from sklearn.tree import plot_tree

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

%matplotlib inline

import matplotlib.pyplot as plt  

import seaborn as sns
train = pd.read_csv("/kaggle/input/data-science-london-scikit-learn/train.csv", header = None)

train_labels = pd.read_csv("/kaggle/input/data-science-london-scikit-learn/trainLabels.csv", header = None)

test = pd.read_csv("/kaggle/input/data-science-london-scikit-learn/test.csv", header = None)
train.describe()
train_labels.describe()
test.describe()
corrmat = train.corr()

plt.subplots(figsize=(12,9))

sns.heatmap(corrmat, vmax=0.9, square=True, center = 0, cmap = 'viridis')
X = train

y = train_labels



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgbc = XGBClassifier()

xgbc.fit(X_train,y_train)

y_pred = xgbc.predict(X_test)



#rf = RandomForestClassifier(max_depth=6, n_estimators = 2000)

#rf.fit(X_train,y_train)



#y_pred = rf.predict(X_test)



print(accuracy_score(y_test, y_pred))
X_output = test

#y_output_pred = rf.predict(X_output)

y_output_pred = xgbc.predict(X_output)



output = pd.DataFrame(columns = ['Id', 'Solution'])

output['Solution'] = y_output_pred

output['Id'] = list(range(1,9001))

print(output.tail())

output.to_csv('output.csv', index=False)