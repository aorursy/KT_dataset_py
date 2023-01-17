# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%matplotlib inline 

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score, GridSearchCV, RandomizedSearchCV

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.metrics import mean_squared_error

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from scipy import stats
wine_df = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

wine_df.head()
wine_df.shape
wine_df.info()
wine_df.describe()
wine_df.hist(bins=50, figsize=(20,15))

plt.show()
# Quality is transformed to categorical attribute, less than 7 is 'Not Good', 7 or greater is 'Good'

wine_df['quality_cat'] = wine_df['quality'].apply(lambda x: 'Good' if x >= 7 else 'Not Good')

wine_df.head()
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(wine_df, wine_df['quality_cat']):

    strat_train_set = wine_df.loc[train_index]

    strat_test_set = wine_df.loc[test_index]
strat_test_set['quality_cat'].value_counts()/len(strat_test_set)
wine_df['quality_cat'].value_counts()/len(wine_df)
strat_train_set.head()
wine = strat_train_set.copy()

wine.head()
corr_matrix = wine.corr()
corr_matrix['quality'].sort_values(ascending=False)
top_corr_attr = ['quality', 'alcohol', 'sulphates', 'volatile acidity']

pd.plotting.scatter_matrix(wine[top_corr_attr], figsize=(20,8))
wine = strat_train_set.drop('quality', axis=1)

wine_labels = strat_train_set['quality'].copy()
lin_reg = LinearRegression()

lin_reg.fit(wine, wine_labels)
some_data = wine.iloc[:5]

some_labels = wine_labels.iloc[:5]

print('Predictions:', lin_reg.predict(some_data))

print('Labels:', list(some_labels))
lin_predictions = lin_reg.predict(wine)

lin_mse = mean_squared_error(wine_labels, lin_predictions)

lin_rmse = np.sqrt(lin_mse)

lin_rmse
tree_reg = DecisionTreeRegressor()

tree_reg.fit(wine, wine_labels)

tree_predictions = tree_reg.predict(wine)

tree_mse = mean_squared_error(wine_labels, tree_predictions)

tree_rmse = np.sqrt(tree_mse)

tree_rmse
tree_scores = cross_val_score(tree_reg, wine, wine_labels, scoring='neg_mean_squared_error', cv=10)

tree_rmse_scores = np.sqrt(-tree_scores)
def display_score(scores):

    print('Scores:', scores)

    print('Mean:', scores.mean())

    print('Standard Deviation:', scores.std())
display_score(tree_rmse_scores)
lin_scores = cross_val_score(lin_reg, wine, wine_labels, scoring='neg_mean_squared_error', cv=10)

lin_rmse_scores = np.sqrt(-lin_scores)

display_score(lin_rmse_scores)
param_grid = [

    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},

    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}

]
forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)

grid_search.fit(wine, wine_labels)
grid_search.best_params_
grid_search.best_estimator_
cv_results = grid_search.cv_results_

for mean_score, params in zip(cv_results['mean_test_score'], cv_results['params']):

    print(np.sqrt(-mean_score), params)
cv_rmse = np.sqrt(-cv_results['mean_test_score'])

cv_rmse.min()
logistic_reg = LogisticRegression()

dist = dict(C=stats.uniform(loc=0, scale=4), penalty=['l2', 'l1'])

clf = RandomizedSearchCV(logistic_reg, dist, random_state=0)

logistic_search = clf.fit(wine, wine_labels)

print('Best params:', logistic_search.best_params_)

print('Best Score:', logistic_search.best_score_)
rmse_collection = {

    'Linear Regression With Cross Validation': lin_rmse_scores.mean(),

    'Decision Tree With Cross Validation': tree_rmse_scores.mean(),

    'Random Forest Grid Search': cv_rmse.min(),

    'Logistic Regression Randomized Search': logistic_search.best_score_

}

for k in rmse_collection.keys():

    print(k, ': ', rmse_collection[k])

X_test = strat_test_set.drop('quality', axis=1)

y_test = strat_test_set['quality'].copy()
final_model = grid_search.best_estimator_
final_predictions = final_model.predict(X_test)
final_mse = mean_squared_error(y_test, final_predictions)

final_rmse = np.sqrt(final_mse)

final_rmse
confidence = 0.95

squared_errors = (final_predictions - y_test) ** 2

np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1, loc=squared_errors.mean(), scale=stats.sem(squared_errors)))
rev_final_model = logistic_search.best_estimator_

rev_final_predictions = rev_final_model.predict(X_test)
rev_final_mse = mean_squared_error(y_test, rev_final_predictions)

rev_final_rmse = np.sqrt(rev_final_mse)

rev_final_rmse
rev_squared_errors = (rev_final_predictions - y_test) ** 2

np.sqrt(stats.t.interval(confidence, len(rev_squared_errors) - 1, loc=rev_squared_errors.mean(), scale=stats.sem(rev_squared_errors)))