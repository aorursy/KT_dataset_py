# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# import matplotlib
import matplotlib.pyplot as plt
# import seaborn
import seaborn as sns
# import sklearn
import sklearn

import warnings
warnings.filterwarnings('ignore')

sns.set()

%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
wine = pd.read_csv('../input/winequality-red.csv')
wine.head()
wine.info()
wine.describe()
from sklearn.model_selection import StratifiedShuffleSplit

# split the data based on the wine quality
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(wine, wine["quality"]):
    strat_train_set = wine.loc[train_index]
    strat_test_set = wine.loc[test_index]
strat_train_set.shape
strat_test_set.shape
wine_strat = strat_train_set.copy()
wine_strat.quality.value_counts()
wine_strat.var()
wine_strat.corr()
plt.figure(figsize=(10,5))


plt.scatter(x='free sulfur dioxide', y='total sulfur dioxide', c='quality', data=wine_strat)
plt.legend();
plt.figure(figsize=(10,5))

plt.scatter(x='density', y='fixed acidity', c='quality', data=wine_strat)
plt.legend();
from pandas.plotting import scatter_matrix

attributes = ["density", "fixed acidity", "free sulfur dioxide","total sulfur dioxide"]
scatter_matrix(wine_strat[attributes], figsize=(16, 10));
wine_strat.hist(figsize=(10,10));
wine_strat.alcohol.plot(kind='box',figsize=(10,10));
wine_strat['fixed acidity'].plot(kind='box',figsize=(10,10));
wine_strat['volatile acidity'].plot(kind='box',figsize=(10,10));
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'total sulfur dioxide', data = wine);
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'free sulfur dioxide', data = wine);
fig = plt.figure(figsize = (10,6))

sns.barplot(x = 'quality', y = 'fixed acidity', data = wine);
fig = plt.figure(figsize = (10,6))

wine_quality_density = wine.loc[:,['quality', 'density']]

wine_quality_density['density'] = np.log(wine_quality_density['density'])

sns.barplot(x = 'quality', y = 'density', data = wine_quality_density);
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'citric acid', data = wine);
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'chlorides', data = wine);
fig = plt.figure(figsize = (10,6))

wine_quality_ph = wine.loc[:,['quality', 'pH']].copy()

wine_quality_ph['pH'] = np.log(wine_quality_ph['pH'])

sns.barplot(x = 'quality', y = 'pH', data = wine_quality_ph);
wine_train = strat_train_set.drop(columns=['quality'], axis=1).copy()
wine_train_labels = strat_train_set["quality"].copy()
wine_test = strat_test_set.drop(columns=['quality'], axis=1).copy()
wine_test_labels = strat_test_set["quality"].copy()
wine_train['total sulfur dioxide'] = np.log(wine_train['total sulfur dioxide'])
wine_test['total sulfur dioxide'] = np.log(wine_test['total sulfur dioxide'])

wine_train['pH'] = np.log(wine_train['pH'])
wine_test['pH'] = np.log(wine_test['pH'])

wine_train['density'] = np.log(wine_train['density'])
wine_test['density'] = np.log(wine_test['density'])
wine_train.var()
wine_train.head()
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# create StandardScaler
st = StandardScaler()

wine_train_scaled = st.fit_transform(wine_train)
wine_test_scaled = st.fit_transform(wine_test)
lr = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)

lr.fit(wine_train_scaled, wine_train_labels)

lr.score(wine_train_scaled, wine_train_labels)
lr.score(wine_test_scaled, wine_test_labels)
rfc = RandomForestClassifier(n_estimators=10)

rfc.fit(wine_train_scaled, wine_train_labels)

rfc.score(wine_train_scaled, wine_train_labels)
rfc.score(wine_test_scaled, wine_test_labels)
tree = DecisionTreeClassifier()

tree.fit(wine_train, wine_train_labels)

tree.score(wine_train, wine_train_labels)
tree.score(wine_test, wine_test_labels)
# feature importances
importances = tree.feature_importances_
importances
indices = np.argsort(importances)[::-1]

names = [wine_train.columns[i] for i in indices]
names
knc = KNeighborsClassifier()

knc.fit(wine_train_scaled, wine_train_labels)

knc.score(wine_train_scaled, wine_train_labels)
knc.score(wine_test_scaled, wine_test_labels)
# Bad = 0 and good = 1
wine_train_labels = np.where(wine_train_labels > 6.5, 1, 0)
wine_test_labels = np.where(wine_test_labels > 6.5, 1, 0)
wine_train_labels.sum()
wine_test_labels.sum()
lr = LogisticRegression(solver='lbfgs', max_iter=100)

lr.fit(wine_train_scaled, wine_train_labels)

lr.score(wine_train_scaled, wine_train_labels)
lr.score(wine_test_scaled, wine_test_labels)
# overfitting?
from sklearn.metrics import classification_report

y_pred_lr = lr.predict(wine_test_scaled)

print(classification_report(wine_test_labels, y_pred_lr, target_names=['bad', 'good']))
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

accuracy_score(wine_test_labels, y_pred_lr), recall_score(wine_test_labels, y_pred_lr)
svc = SVC()

svc.fit(wine_train_scaled, wine_train_labels)

svc.score(wine_train_scaled, wine_train_labels)
svc.score(wine_test_scaled, wine_test_labels)
y_pred_svc = svc.predict(wine_test_scaled)
print(classification_report(wine_test_labels, y_pred_svc, target_names=['bad', 'good']))
from sklearn.svm import LinearSVC

lsvc = LinearSVC(max_iter=1000)

lsvc.fit(wine_train_scaled, wine_train_labels)

lsvc.score(wine_train_scaled, wine_train_labels)
lsvc.score(wine_test_scaled, wine_test_labels)
y_pred_lsvc = lsvc.predict(wine_test_scaled)
print(classification_report(wine_test_labels, y_pred_lsvc, target_names=['bad', 'good']))
rfc = RandomForestClassifier()

rfc.fit(wine_train_scaled, wine_train_labels)

rfc.score(wine_train_scaled, wine_train_labels)
rfc.score(wine_test_scaled, wine_test_labels)
y_pred_rfc = rfc.predict(wine_test_scaled)
print(classification_report(wine_test_labels, y_pred_rfc, target_names=['bad', 'good']))
tree = DecisionTreeClassifier()

tree.fit(wine_train, wine_train_labels)

tree.score(wine_train, wine_train_labels)
tree.score(wine_test, wine_test_labels)
y_pred_tree = tree.predict(wine_test)
print(classification_report(wine_test_labels, y_pred_tree, target_names=['bad', 'good']))
sgd = SGDClassifier(max_iter=1000, tol=1e-3)

sgd.fit(wine_train_scaled, wine_train_labels)

sgd.score(wine_train_scaled, wine_train_labels)
sgd.score(wine_test_scaled, wine_test_labels)
y_pred_sgd = sgd.predict(wine_test_scaled)
print(classification_report(wine_test_labels, y_pred_sgd, target_names=['bad', 'good']))
knc = KNeighborsClassifier()

knc.fit(wine_train_scaled, wine_train_labels)

knc.score(wine_train_scaled, wine_train_labels)
knc.score(wine_test_scaled, wine_test_labels)
y_pred_knc = knc.predict(wine_test_scaled)
print(classification_report(wine_test_labels, y_pred_knc, target_names=['bad', 'good']))
pipe = Pipeline([('preprocessing', StandardScaler()), ('classifier', SVC())])
from sklearn.model_selection import GridSearchCV

param_grid_svc = [
    {'classifier': [SVC()], 
     'classifier__kernel':['linear', 'rbf'],
     'classifier__gamma': [0.001, 0.01, 0.1, 1, 10, 100],
     'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]
    }
]
grid = GridSearchCV(pipe, param_grid_svc, scoring='accuracy', cv=10)
grid.fit(wine_train, wine_train_labels)
grid.best_score_
grid.best_params_
y_pred_grid_svc = grid.best_estimator_.predict(wine_test)

print(classification_report(wine_test_labels, y_pred_grid_svc, target_names=['bad', 'good']))
accuracy_score(wine_test_labels, y_pred_grid_svc)
param_grid_rfc = [
    {'classifier': [RandomForestClassifier()], 
     'classifier__n_estimators': np.arange(2,20),
     'classifier__max_leaf_nodes' : np.arange(2,10),
     'classifier__max_depth': np.arange(2,10),
    }
]
grid_rfc = GridSearchCV(pipe, param_grid_rfc, scoring='accuracy', cv=10)
grid_rfc.fit(wine_train, wine_train_labels);
grid_rfc.best_params_
grid_rfc.best_score_
y_pred_grid_rfc = grid_rfc.best_estimator_.predict(wine_test)

print(classification_report(wine_test_labels, y_pred_grid_rfc, target_names=['bad', 'good']))
accuracy_score(wine_test_labels, y_pred_grid_rfc)
param_grid_dtc = [
    {'classifier': [DecisionTreeClassifier()], 
     'preprocessing': [None],  
     'classifier__criterion': ['gini', 'entropy'],
     'classifier__max_depth': np.arange(2,20)
    }
]
grid_dtc = GridSearchCV(pipe, param_grid_dtc, scoring='accuracy', cv=10)
grid_dtc.fit(wine_train, wine_train_labels);
grid_dtc.best_score_
grid_dtc.best_params_
y_pred_test_dtc = grid_dtc.best_estimator_.predict(wine_test)

print(classification_report(wine_test_labels, y_pred_grid_rfc, target_names=['bad', 'good']))
accuracy_score(wine_test_labels, y_pred_test_dtc)
