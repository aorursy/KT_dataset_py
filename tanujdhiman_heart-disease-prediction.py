# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('../input/heart-disease-uci/heart.csv')
# Look at our dataset
data.head()
data.target.unique()
# info method tells about the datatypes of all columns in the dataset.
data.info()
# It is a small dataset as you can see just 303 values
data.shape
data.describe()
X = data.iloc[:, 0:13].values
y = data.iloc[:, 13].values
X
y
from sklearn.ensemble import ExtraTreesClassifier
# n_estimator is parameter which try through all the dataset n times
clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
clf.fit(X, y)
important_features = clf.feature_importances_
important_features.max()
important_features
columns = data.columns
cols = columns[0:13]
cols
plt.figure(figsize = (15, 5))
plt.bar(cols, important_features, color = "green")

plt.xlabel("Name of Columns")
plt.ylabel("Columns Importances")

plt.title("Important Features Graph")
plt.show()
print(important_features)
print(cols)
important_columns = []
for i in range(len(important_features)):
    if(important_features[i] >= 0.08):
        important_columns.append(cols[i])
important_columns
X_new = data.iloc[:, [2, 7, 8, 9, 11, 12]].values
X_new
y
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.1)
X_train
y_train
X_test
y_test
from sklearn.tree import DecisionTreeClassifier as dtc
classifier = dtc(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, 
                 min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, 
                 random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, 
                 min_impurity_split=None, class_weight=None)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_pred
y_test
from sklearn.model_selection import cross_val_score
scores = cross_val_score(classifier, X_train, y_train, cv=5)
scores
classifier.score(X_test, y_test)
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
conf_matrix
print('Precision: %.3f' % precision_score(y_test, y_pred))
print('Recall: %.3f' % recall_score(y_test, y_pred))
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
print('F1 Score: %.3f' % f1_score(y_test, y_pred))