import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))
df_train = pd.read_csv('../input/train.csv')
df_test  = pd.read_csv('../input/test.csv')
df_train.head()
# Prepare data set
X = df_train.drop(columns='Survived')
X_test = df_test.copy()
PassengerId_test = X_test['PassengerId']
y = df_train['Survived']

# Fill missing values with mean values for Age column
X['Age'].fillna(X['Age'].mean(), inplace=True)
X_test['Age'].fillna(X['Age'].mean(), inplace=True)
X_test['Fare'].fillna(X['Fare'].mean(), inplace=True)

# Drop columns
X.drop(columns=['Name', 'Ticket', 'PassengerId'], inplace=True)
X_test.drop(columns=['Name', 'Ticket', 'PassengerId'], inplace=True)

# Get list of numeric columns
numeric_columns = [col for col in X.columns if X.dtypes[col] != "object"]
from pandas.plotting import scatter_matrix
scatter_matrix(X[numeric_columns], alpha=0.2, figsize=(8, 8));
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=1)
model.fit(X[numeric_columns], y)

# Out-of-bag prediction score
model.oob_score_
features = pd.Series(model.feature_importances_, index=X[numeric_columns].columns).sort_values()
features.plot(kind='barh', color='C0', title='Feature Importance');
# Get only first letter for each Cabin 
X['Cabin'] = X['Cabin'].apply(lambda x: 'NONE' if (isinstance(x, float) and np.isnan(x)) else x[0])
X_test['Cabin'] = X_test['Cabin'].apply(lambda x: 'NONE' if isinstance(x, float) and np.isnan(x) else x[0])
# Convert categorical columns to one-hot-encoded columns
for col in ['Cabin', 'Embarked', 'Sex']:
    X[col].fillna('NONE', inplace=True)
    one_hot_cols = pd.get_dummies(X[col], prefix=col)
    X = pd.concat([X, one_hot_cols], axis=1).drop(columns=col)
    
    X_test[col].fillna('NONE', inplace=True)
    one_hot_cols = pd.get_dummies(X_test[col], prefix=col)
    X_test = pd.concat([X_test, one_hot_cols], axis=1).drop(columns=col)

# Add missing columns for test data
for col in list(set(X.columns) - set(X_test.columns)):
    n_rows = X_test.shape[0]
    X_test[col] = np.zeros((n_rows))

X.head()
model = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=1)
model.fit(X, y)

# Out-of-bag prediction score
model.oob_score_
features = pd.Series(model.feature_importances_, index=X.columns).sort_values()
features.plot(kind='barh', color='C0', title='Feature Importance');
from sklearn.model_selection import GridSearchCV

clf = RandomForestClassifier()

n_estimators = [20, 50, 100, 200, 500, 1000, 2000]
param_grid = dict(n_estimators=n_estimators)

grid = GridSearchCV(clf, param_grid, cv=10, scoring='accuracy')
grid.fit(X, y)

# Get results
results = grid.cv_results_

# Code adapted from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html
for i in range(1, 6):
    candidates = np.flatnonzero(results['rank_test_score'] == i)
    for candidate in candidates:
        print("Model with rank: {}".format(i))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
        print("Parameters: {}".format(results['params'][candidate]))
        print()

x_pos = range(len(n_estimators))
plt.plot(grid.cv_results_['mean_test_score'])
plt.xticks(x_pos, n_estimators)
plt.xlabel('n_estimator')
plt.ylabel('Cross-Validated Accuracy');
from sklearn.model_selection import RandomizedSearchCV

clf = RandomForestClassifier()

n_estimators = [20, 50, 100, 200, 500, 1000, 2000]
min_samples_leaf = [1, 2, 3, 4, 5, 6, 8, 9, 10]
max_features = ['auto', None, 'sqrt', 'log2', 0.2, 0.8]

param_grid = dict(n_estimators=n_estimators, 
                  min_samples_leaf=min_samples_leaf, 
                  max_features=max_features)

random_search = RandomizedSearchCV(clf, param_grid, n_iter=20)
random_search.fit(X, y)

# Get results
results = random_search.cv_results_

# Code adapted from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html
for i in range(1, 6):
    candidates = np.flatnonzero(results['rank_test_score'] == i)
    for candidate in candidates:
        print("Model with rank: {}".format(i))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
        print("Parameters: {}".format(results['params'][candidate]))
        print()
y_pred = random_search.predict(X_test)

df_submission = pd.DataFrame({'PassengerId': PassengerId_test, 'Survived': y_pred}).set_index('PassengerId')
df_submission.to_csv('submission.csv')