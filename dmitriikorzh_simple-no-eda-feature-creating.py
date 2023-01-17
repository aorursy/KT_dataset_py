import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import Ridge
X_train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
X_test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
ss = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
X_train.head()
q=X_train.isna().sum()
q = q[q>0]
print(q)
q1=X_test.isna().sum()
q1 = q1[q1>0]
print(q1)
y_train = X_train["SalePrice"]

X_train.set_index("Id", inplace=True)
X_train.drop("SalePrice", axis=1, inplace=True)

X_test.set_index("Id", inplace=True)

numerical_features = [feature for feature in X_train if X_train[feature].dtype != "object"]
categorical_features = [feature for feature in X_train if X_train[feature].dtype == "object" and X_train[feature].isna().mean() < 0.3 and
                        X_train[feature].nunique() < 20]




X_train = X_train[numerical_features + categorical_features]
X_test = X_test[numerical_features + categorical_features]
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
preprocessor = make_column_transformer((SimpleImputer(strategy="mean"), numerical_features),
    (make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), categorical_features) )
ridge_reg = Ridge()


pipe1 = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', ridge_reg)])

param_grid = {'regressor__alpha': np.logspace(-6,1,40),}
#parameters['feat_select__k'] = [5, 10]

CV = GridSearchCV(pipe1, param_grid, n_jobs= 1)

CV.fit(X_train, y_train) 
y_test_pred_Ridge = CV.best_estimator_.predict(X_test)
y_test_pred_Ridge
y_train_pred_Ridge = CV.best_estimator_.predict(X_train)
print('Train MSE: ', np.sqrt(mse(y_train, y_train_pred)))

from sklearn.ensemble import RandomForestRegressor
rfreg=RandomForestRegressor(random_state=42)
pipe2 = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', rfreg)])
param_grid2 = {'regressor__n_estimators': [10, 20, 50, 100, 200],
              'regressor__max_depth': np.arange(2, 15)}

search = GridSearchCV(pipe2, param_grid2, n_jobs=-1)
search.fit(X_train, y_train)
y_test_pred_RF = search.best_estimator_.predict(X_test)
y_test_pred_RF
ss['SalePrice'] = y_test_pred_RF
ss.to_csv("../input/submission_RF.csv", index = False)


pipe2 = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', rfreg)])
param_grid2 = {'regressor__n_estimators': [10, 20, 50, 100, 200],
              'regressor__max_depth': np.arange(2, 15)}

search = GridSearchCV(pipe2, param_grid2, n_jobs=-1)
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

xg_reg = xgb.XGBRegressor(n_estimators=100)

pipe3 = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', xg_reg)])

param_grid3 = {'regressor__n_estimators': [10, 20, 50, 100, 200, 300, 4000],
              'regressor__max_depth': np.arange(2, 15)}


search3 = GridSearchCV(pipe3, param_grid3, n_jobs=-1)
search3.fit(X_train,y_train)
y_test_pred_XGB = search3.best_estimator_.predict(X_test)
ss['SalePrice'] = y_test_pred_XGB
ss.to_csv("../input/submission_XGB.csv", index = False)
def make_model(model):
    return make_pipeline(preprocessor, model)

models = [make_model(model) for model in [RandomForestRegressor(), GradientBoostingRegressor(),KernelRidge(), XGBRegressor()]]

for model in models:
    model.fit(X_train, y_train)


y_pred = sum(model.predict(X_test) for model in models)/len(models)

ss['SalePrice'] = y_pred
ss.head()
ss.to_csv("../input/submission2.csv", index = False)
