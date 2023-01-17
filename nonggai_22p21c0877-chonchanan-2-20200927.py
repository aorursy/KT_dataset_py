# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
%matplotlib inline
train_set = pd.read_csv('../input/titanic/train.csv', index_col='PassengerId')
print(train_set.shape)
train_set.head()
train_set.info()
train_set.isnull().sum()
print('Unique_Survived: ', train_set['Survived'].unique())
print('Unique_Pclass: ', train_set['Pclass'].unique())
print('Unique_Sex: ', train_set['Sex'].unique())
print('Unique_SibSp: ', train_set['SibSp'].unique())
print('Unique_Parch: ', train_set['Parch'].unique())
print('Unique_Cabin: ', train_set['Cabin'].unique())
print('Unique_Embarked: ', train_set['Embarked'].unique())
train_set.describe()
xvals = np.arange(len(train_set['Survived'].value_counts()))
yvals = list(train_set['Survived'].value_counts())

counts, bins, _ = plt.hist(train_set['Survived'], bins=2)

for n, b in zip(counts, bins):
    if n > 0:
        plt.gca().text(b + 0.2, n, str(n))  # +0.1 to center text
plt.show()
train_set.dropna(subset=['Embarked'], inplace=True)
train_set
train_label = train_set['Survived']
train_label.head()
train_set.drop(['Survived', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
train_set.head()
train_set.interpolate(method='linear', inplace=True)
train_set.isnull().sum()
train_set['Sex'].replace('female', 0, inplace=True)
train_set['Sex'].replace('male', 1, inplace=True)
train_set.head()
Pclass_train = pd.get_dummies(train_set.Pclass, prefix='Pclass')
Pclass_train
Embarked_train = pd.get_dummies(train_set.Embarked, prefix='Embarked')
Embarked_train
train_set = pd.concat([train_set, Pclass_train, Embarked_train], axis=1)
train_set.drop(['Pclass', 'Embarked'], axis=1, inplace=True)
train_set
train_set = (train_set-train_set.min())/(train_set.max()-train_set.min())
train_set
train_set.describe()
corr = train_set.corr()

f, ax = plt.subplots(figsize=(12, 12))
ax = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True)
X_train, X_test, y_train, y_test = train_test_split(train_set, train_label, test_size=0.2, random_state=42)
X_train
params_tree = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}
grid_search_cv_tree = GridSearchCV(DecisionTreeClassifier(random_state=42), params_tree, verbose=1, cv=5)
grid_search_cv_tree.fit(X_train, y_train)
print('Best estimator for Decision tree classifier: ', grid_search_cv_tree.best_estimator_)
fig, ax = plt.subplots(figsize=(30, 30))
tree.plot_tree(grid_search_cv_tree.best_estimator_, ax=ax)
plt.show()
y_predict_tree = grid_search_cv_tree.best_estimator_.predict(X_test)
print(classification_report(y_test, y_predict_tree, target_names=['No', 'Yes']))
params_bay = {}
grid_search_cv_bay = GridSearchCV(GaussianNB(), param_grid=params_bay, n_jobs=-1, cv=5, verbose=5)
grid_search_cv_bay.fit(X_train, y_train)
print('Best estimator for Gaussian naive bayes: ', grid_search_cv_bay.best_estimator_)
y_predict_bay = grid_search_cv_bay.best_estimator_.predict(X_test)
print(classification_report(y_test, y_predict_bay, target_names=['No', 'Yes']))
params_nn = {'hidden_layer_sizes': [(10,30,10),(20,)], 'activation': ['tanh', 'relu'], 'solver': ['sgd', 'adam'], 'alpha': [0.0001, 0.05], 'learning_rate': ['constant','adaptive'],}
grid_search_cv_nn = GridSearchCV(MLPClassifier(max_iter=200), param_grid=params_nn, n_jobs=-1, cv=5, verbose=5)
grid_search_cv_nn.fit(X_train, y_train)
print('Best estimator for Multi-layer perceptron classifier: ', grid_search_cv_nn.best_estimator_)
y_predict_nn = grid_search_cv_nn.best_estimator_.predict(X_test)
print(classification_report(y_test, y_predict_nn, target_names=['No', 'Yes']))
