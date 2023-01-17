# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

print(train.columns.values)
print(test.columns.values)
X = train.drop(['PassengerId', 'Name', 'Survived', 'Ticket', 'Cabin'], axis=1)
y = train['Survived']
Xs = pd.get_dummies(X, columns=['Pclass', 'Sex', 'Embarked'])
Xs['Age'].fillna(value=X['Age'].mean(), inplace=True)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

%matplotlib inline
scaler = StandardScaler()
X_scale = scaler.fit_transform(Xs[['Age', 'Fare']])
print(X_scale.shape, len(Xs['Age']))

Xs['Age'] = X_scale[:, 0]
Xs['Fare'] = X_scale[:, 1]
Xs

pipeline = Pipeline((
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2))
))

Xn = pipeline.fit_transform(Xs)

colors = ['go', 'bs']
plt.figure(figsize=(16, 12))
for v in np.unique(y):
    index = np.where(y == v)
    plt.plot(Xn[index, 0], Xn[index, 1], colors[v], alpha=0.5)
X_train, X_test, y_train, y_test = train_test_split(Xs, y, train_size=0.8, random_state=18)
def evaluate(clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(accuracy_score(y_pred, y_test))
tree_clf = DecisionTreeClassifier(random_state=18)
evaluate(tree_clf)
params = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 2, 4, 6],
    'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
    'min_samples_leaf': [1, 2, 3, 4, 5],
}
grid_clf = GridSearchCV(tree_clf, params, verbose=2, n_jobs=-1)

grid_clf.fit(X_train, y_train)
evaluate(grid_clf.best_estimator_)
tree_clf.set_params(**grid_clf.best_params_)
rf_clf = RandomForestClassifier(random_state=18)
evaluate(rf_clf)
# params = {
#     'n_estimators': [10, 100, 200, 500, 1000],
#     'criterion': ['gini', 'entropy'],
#     'max_features': ['auto', 'sqrt', 'log2', None],
#     'max_depth': [None, 2, 4, 6],
#     'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
#     'min_samples_leaf': [1, 2, 3, 4, 5],
# }

# grid_clf = GridSearchCV(rf_clf, params, verbose=2, n_jobs=-1)

# grid_clf.fit(X_train, y_train)
# evaluate(grid_clf.best_estimator_)

# rf_clf.set_params(**grid_clf.best_params_)
rf_clf.set_params(
    min_samples_split=8, 
    min_samples_leaf=3, 
    max_depth=None, 
    max_features=None, 
    n_estimators=100,
    n_jobs=-1,
    random_state=18
)

evaluate(rf_clf)
svc_clf = SVC()

evaluate(svc_clf)
params = {
    'C': [0.001, 0.1, 1.0, 2.0, 5.0, 10.0],
    'kernel': ['linear', 'rbf'],
    'gamma': ['auto', 0.1, 0.5, 1.0, 2.0],
}

grid_svc_clf = GridSearchCV(svc_clf, params, n_jobs=-1, verbose=2)

grid_svc_clf.fit(X_train, y_train)

evaluate(grid_svc_clf.best_estimator_)

svc_clf.set_params(**grid_svc_clf.best_params_)
Xt = test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

Xt = pd.get_dummies(Xt, columns=['Pclass', 'Sex', 'Embarked'])
Xt['Age'].fillna(value=X['Age'].mean(), inplace=True)
Xt['Fare'].fillna(value=X['Fare'].mean(), inplace=True)

Xtscale = scaler.fit_transform(Xt[['Age', 'Fare']])

Xt['Age'] = Xtscale[:, 0]
Xt['Fare'] = Xtscale[:, 1]
Xt
y_pred = rf_clf.predict(Xt)
test['Survived'] = y_pred

test
test[['PassengerId', 'Survived']].to_csv('titanic.csv', index=False)
with open('titanic.csv') as f:
    print(f.read())