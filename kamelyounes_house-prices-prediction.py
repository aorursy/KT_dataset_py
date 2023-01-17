# python libraries
import pandas as pd
import numpy as np

# regressors
import xgboost
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# pre-processing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

data = pd.read_csv("../input/home-data-for-ml-course/train.csv")
data.head(10)
print(data.isnull().sum())
for key, values in data.iteritems():
    if (pd.api.types.is_numeric_dtype(data[key])):
        data[key].fillna(value= data[key].mean(), inplace=True)
    else :
        data[key].fillna(value= "Missing", inplace=True) 

    
mising_data = pd.Series(data.isnull().sum())     

one_hot_enc = LabelEncoder()
for key, values in data.iteritems():
    if (pd.api.types.is_string_dtype(data[key])):        
           data[key] = one_hot_enc.fit_transform(data[key])


scaler = StandardScaler()
scaler.fit(data)
scaler.transform(data)

X = data.drop('SalePrice', 1)
y = data.SalePrice

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# linear regression grid search
grid_search_linear_reg = GridSearchCV(LinearRegression(), { 'fit_intercept': [True, False],
                                                        'normalize': [True, False], 
                                                        'copy_X': [True, False] 
                                                        }, cv=5)
grid_search_linear_reg.fit(X_train, y_train)
print("linear reg score :", grid_search_linear_reg.best_score_)

# decision tree grid search
decision_tree_param_grid = {'criterion': ['mse', 'mae'],
              'min_samples_split': [10, 20, 40],
              'max_depth': [2, 6, 8],
              'min_samples_leaf': [20, 40, 100],
              'max_leaf_nodes': [5, 20, 100],
              }

grid_search_decision_trees = GridSearchCV(DecisionTreeRegressor(), decision_tree_param_grid, cv=5)
grid_search_decision_trees.fit(X_train, y_train)
print("decision trees score :", grid_search_decision_trees.best_score_)

# xgboost grid search
xgb_param_grid = {
    'n_estimators': [100, 500, 900, 1100, 1500],
    'max_depth': [2,3,5,10,15],
    'learning_rate': [0.05, 0.1, 0.15, 0.2],
    'min_child_weight': [1,2,3,4],
    'booster': ['gbtree','gblinear'],
    'base_score': [0.25, 0.5, 0.75, 1]
}
grid_search_xgb = RandomizedSearchCV(xgboost.XGBRegressor(), param_distributions = xgb_param_grid,
                              cv=5, n_iter=50,
                              scoring = 'neg_mean_absolute_error', n_jobs = 4,
                              verbose = 5,
                              return_train_score = True,
                              random_state = 42)
grid_search_xgb.fit(X_train, y_train)
xgb_best_estimator = grid_search_xgb.best_estimator_
print("xgb score :",cross_val_score(xgb_best_estimator, X_train, y_train, cv=5).mean())
xgb_best_estimator.fit(X_train, y_train)
xgb_best_estimator.score(X_test, y_test)
