# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/../input/churn-in-telecoms-dataset/bigml_59c28831336c6604c800002a.csv")
df
df = df.drop(['state', 'phone number'], axis='columns')
df['area code'] = df['area code'].map({408: 0, 415: 1, 510: 2})
df['voice mail plan'] = df['voice mail plan'].map( {"no": 0,"yes": 1} )
df['international plan'] = df['international plan'].map( {"no": 0,"yes": 1} )
df['churn'] = df['churn'].map( {False: 0, True: 1 })
df.head()
scaler = StandardScaler()
Xs = df.drop('churn', axis='columns')
X = scaler.fit_transform(df.drop('churn', axis='columns'))
Y = df['churn']
X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size = 0.25, random_state=42)
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_train, y_train)
res = tree.predict(X_valid)
print(f1_score(y_valid, res))
kfold = KFold(n_splits = 5, shuffle = True, random_state = 42)

params = {'max_depth': np.arange(2, 30)}
grid = GridSearchCV(tree, params, cv=kfold, scoring='f1')
grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.best_score_)
sns.lineplot(params['max_depth'], grid.cv_results_['mean_test_score'])
tree = DecisionTreeClassifier(max_depth=6, random_state=42)
params = {'min_samples_split': np.arange(2, 35)}
grid = GridSearchCV(tree, params, cv=kfold, scoring='f1')
grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.best_score_)
sns.lineplot(params['min_samples_split'], grid.cv_results_['mean_test_score'])
tree = DecisionTreeClassifier(max_depth=5, min_samples_split=9, random_state=42)
tree.fit(X_train, y_train)
params = {'min_samples_leaf': np.arange(1, 40)}
grid = GridSearchCV(tree, params, cv=kfold, scoring='f1')
grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.best_score_)
sns.lineplot(params['min_samples_leaf'], grid.cv_results_['mean_test_score'])
tree = DecisionTreeClassifier(max_depth=5, min_samples_split=9, min_samples_leaf= 12, random_state=42)
tree.fit(X_train, y_train)
params = {'max_features': np.arange(1, 18)}
grid = GridSearchCV(tree, params, cv=kfold, scoring='f1')
grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.best_score_)
sns.lineplot(params['max_features'], grid.cv_results_['mean_test_score'])
tree = DecisionTreeClassifier(max_depth=5, min_samples_split=9, min_samples_leaf=12, max_features=17, random_state=42)
tree.fit(X_train, y_train)
export_graphviz(tree, out_file='tree.dot', feature_names=Xs.columns)
features = list(enumerate(Xs.columns))

inds = np.argsort(tree.feature_importances_)[::-1]

for i in range(num_to_plot):
    print(features[inds[i]], tree.feature_importances_[inds[i]])
tree = RandomForestClassifier(random_state=42)
tree.fit(X_train, y_train)
res = tree.predict(X_valid)
print(f1_score(y_valid, res))
params = {"n_estimators":[50, 75, 100, 150, 200, 250, 400, 500, 600]}
grid = GridSearchCV(tree, params, cv=kfold, scoring='f1')
grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.best_score_)
sns.lineplot(params['n_estimators'], grid.cv_results_['mean_test_score'])
tree = RandomForestClassifier(random_state=42, n_estimators=200)
params = {'max_depth': np.arange(3, 10)}
grid = GridSearchCV(tree, params, cv=kfold, scoring='f1')
grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.best_score_)
sns.lineplot(params['max_depth'], grid.cv_results_['mean_test_score'])
tree = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=9)
params = {'min_samples_split': np.arange(2, 20)}
grid = GridSearchCV(tree, params, cv=kfold, scoring='f1')
grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.best_score_)
sns.lineplot(params['min_samples_split'], grid.cv_results_['mean_test_score'])
tree = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=9, min_samples_split=5)
params = {'min_samples_leaf': np.arange(4, 18)}
grid = GridSearchCV(tree, params, cv=kfold, scoring='f1')
grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.best_score_)
sns.lineplot(params['min_samples_leaf'], grid.cv_results_['mean_test_score'])
tree = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=9, min_samples_split=5, min_samples_leaf=4)
params = {'max_features': np.arange(2, 18)}
grid = GridSearchCV(tree, params, cv=kfold, scoring='f1')
grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.best_score_)
sns.lineplot(params['max_features'], grid.cv_results_['mean_test_score'])
tree = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=9, min_samples_split=5, min_samples_leaf=4, max_features=6)
tree.fit(X_train, y_train)
features = list(enumerate(Xs.columns))

inds = np.argsort(tree.feature_importances_)[::-1]

for i in range(num_to_plot):
    print(features[inds[i]], tree.feature_importances_[inds[i]])
"""
(7, 'total day charge') 0.20416138957402968
(17, 'customer service calls') 0.1538188690666516
(15, 'total intl calls') 0.12356276037086658
(2, 'international plan') 0.10834663405628774
(5, 'total day minutes') 0.10284871878253567
(16, 'total intl charge') 0.08853406506323773
(8, 'total eve minutes') 0.07337652164827653
(4, 'number vmail messages') 0.06464745348737191
(10, 'total eve charge') 0.06451476217305137
(13, 'total night charge') 0.0078098854085267904
"""