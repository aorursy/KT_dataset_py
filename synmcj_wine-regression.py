import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

data_path = "../input/winequality-red.csv"

# checking data
data = pd.read_csv(data_path)
data.head()
data.describe()
%matplotlib inline
import matplotlib.pyplot as plt

data.hist(bins=50, figsize=(20, 15))
plt.show()
from sklearn.model_selection import StratifiedShuffleSplit

# splitting to test and train data
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data, data["quality"]):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]
    
data = strat_train_set.copy()
data = strat_train_set.drop("quality", axis=1)
data_labels = strat_train_set["quality"].copy()
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# setting up pipeline for preparing data
pipeline = Pipeline([
    ('std_scaler', StandardScaler()),
])

data_prepared = pipeline.fit_transform(data)
data_prepared.shape
# method for displaying evaluation scores
def display_scores(scores):
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# training linear regression using cross validation
lin_reg = LinearRegression()
lin_reg.fit(data_prepared, data_labels)

lin_scores = cross_val_score(lin_reg, data_prepared, 
    data_labels, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

# training decision tree regressor using cross validation
tree_reg = DecisionTreeRegressor()
tree_reg.fit(data_prepared, data_labels)

scores = cross_val_score(tree_reg, data_prepared, 
    data_labels, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
display_scores(tree_rmse_scores)
from sklearn.ensemble import RandomForestRegressor

# training random forest regressor using cross validation
forest_reg = RandomForestRegressor()
forest_reg.fit(data_prepared, data_labels)

scores = cross_val_score(forest_reg, data_prepared, 
    data_labels, scoring="neg_mean_squared_error", cv=10)
forest_rmse = np.sqrt(-scores)
display_scores(forest_rmse)
from sklearn.model_selection import GridSearchCV

# fine tuning best model (RandomForestRegressor)
param_grid = [
    {'n_estimators': [30, 50, 70, 90], 'max_features': [4, 6, 8, 10, 11]},
    {'bootstrap': [False], 'n_estimators': [10, 30, 50], 'max_features': [6, 8, 11]},
]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5, 
    scoring='neg_mean_squared_error')
grid_search.fit(data_prepared, data_labels)

print(grid_search.best_params_)
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
# checking out feature importance
feature_importances = grid_search.best_estimator_.feature_importances_
attributes = list(data)
sorted(zip(feature_importances, attributes), reverse=True)
# testing final model
final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("quality", axis=1)
y_test = strat_test_set["quality"].copy()

X_test_prepared = pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse