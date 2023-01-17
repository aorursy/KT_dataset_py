import numpy as np
import pandas as pd
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
test["casual"] = np.nan
test["registered"] = np.nan
test["count"] = np.nan

train["is_train"] = 1
test["is_train"] = 0
full = pd.concat([train, test], axis = 0)
full["datetime"] = full["datetime"].astype('datetime64')
full["season"] = full["season"].astype('category')
full["holiday"] = full["holiday"].astype('bool')
full["workingday"] = full["workingday"].astype('bool')
full["weather"] = full["weather"].astype('category')
full["is_train"] = full["is_train"].astype('bool')
full["hour"] = full["datetime"].dt.hour.astype('category')
full["month"] = full["datetime"].dt.month.astype('category')
full["year"] = full["datetime"].dt.year.astype('category')
full = full.set_index('datetime')
full = full.drop(["season", "atemp", "registered", "casual"], axis = 1)
factor_variables = ["weather", "hour", "month", "year"]
full_dummies = pd.get_dummies(full[factor_variables])
non_factor_variables = ["holiday", "workingday", "temp", "humidity", "windspeed", "count", "is_train"]
full_no_dummies = full[non_factor_variables]
full = pd.concat([full_no_dummies, full_dummies], axis = 1)
train = full[full["is_train"] == 1]
test = full[full["is_train"] == 0]
X_train = train.drop(["is_train", "count", "year_2012"], axis = 1)
y_train = train["count"]

X_test = test.drop(["is_train", "count", "year_2012"], axis = 1)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
rfr = RandomForestRegressor()
param_grid = [
  {'max_depth': [20, 30, 40, 50],
   'min_samples_leaf': [1, 5, 10, 20],
   'n_estimators':[30, 50, 70, 90],
   'random_state':[42]}]
rfr = GridSearchCV(rfr, param_grid, cv = 5, scoring = 'neg_mean_squared_log_error')
rfr.fit(X_train, y_train)
rfr.best_score_
rfr.best_params_
rfr = RandomForestRegressor()
param_grid = [
  {'max_depth': [25, 30, 35],
   'min_samples_leaf': [1, 3, 5],
   'n_estimators':[40, 50, 60],
   'random_state':[42]}]
rfr = GridSearchCV(rfr, param_grid, cv = 5, scoring = 'neg_mean_squared_log_error')
rfr.fit(X_train, y_train)
rfr.best_score_
rfr.best_params_
rfr = RandomForestRegressor()
param_grid = [
  {'max_depth': [28, 30, 32],
   'min_samples_leaf': [1, 2],
   'n_estimators':[55, 60, 65],
   'random_state':[42]}]
rfr = GridSearchCV(rfr, param_grid, cv = 5, scoring = 'neg_mean_squared_log_error')
rfr.fit(X_train, y_train)
rfr.best_score_
rfr.best_params_
rfr = RandomForestRegressor()
param_grid = [
  {'max_depth': [29, 30, 31],
   'min_samples_leaf': [1],
   'n_estimators':[50, 55, 60],
   'random_state':[42]}]
rfr = GridSearchCV(rfr, param_grid, cv = 5, scoring = 'neg_mean_squared_log_error')
rfr.fit(X_train, y_train)
rfr.best_score_
rfr.best_params_
rfr = RandomForestRegressor()
param_grid = [
  {'max_depth': [30],
   'min_samples_leaf': [1],
   'n_estimators':[52, 53, 54, 55, 56, 57, 58],
   'random_state':[42]}]
rfr = GridSearchCV(rfr, param_grid, cv = 5, scoring = 'neg_mean_squared_log_error')
rfr.fit(X_train, y_train)
rfr.best_score_
rfr.best_params_
prediction = rfr.predict(X_test)
result = pd.DataFrame(test.index).assign(count = prediction)
result.to_csv('output_pred.csv', index=False)