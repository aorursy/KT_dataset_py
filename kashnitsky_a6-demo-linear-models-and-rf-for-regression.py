import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.metrics.regression import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression, LassoCV, Lasso
from sklearn.ensemble import RandomForestRegressor
data = pd.read_csv('../input/winequality-white.csv')
data.head()
data.info()
# y = None # you code here

# X_train, X_holdout, y_train, y_holdout = train_test_split # you code here
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform # you code here
# X_holdout_scaled = scaler.transform # you code here
# linreg = # you code here
# linreg.fit # you code here
# print("Mean squared error (train): %.3f" % # you code here
# print("Mean squared error (test): %.3f" % # you code here
# linreg_coef = pd.DataFrame # you code here
# linreg_coef.sort_values # you code here
# lasso1 = Lasso # you code here
# lasso1.fit # you code here
# lasso1_coef = pd.DataFrame # you code here
# lasso1_coef.sort_values # you code here
# alphas = np.logspace(-6, 2, 200)
# lasso_cv = LassoCV # you code here
# lasso_cv.fit # you code here
# lasso_cv.alpha_
# lasso_cv_coef = pd.DataFrame # you code here
# lasso_cv_coef.sort_values # you code here
# print("Mean squared error (train): %.3f" % # you code here
# print("Mean squared error (test): %.3f" % # you code here
# forest = RandomForestRegressor # you code here
# forest.fit # you code here
# print("Mean squared error (train): %.3f" % # you code here
# print("Mean squared error (cv): %.3f" % # you code here
# print("Mean squared error (test): %.3f" % # you code here
# forest_params = {'max_depth': list(range(10, 25)), 
#                  'min_samples_leaf': list(range(1, 8)),
#                  'max_features': list(range(6,12))}

# locally_best_forest = GridSearchCV # you code here
# locally_best_forest.fit # you code here
# locally_best_forest.best_params_, locally_best_forest.best_score_
# print("Mean squared error (cv): %.3f" % # you code here
# print("Mean squared error (test): %.3f" % # you code here
# rf_importance = pd.DataFrame # you code here
# rf_importance.sort_values # you code here