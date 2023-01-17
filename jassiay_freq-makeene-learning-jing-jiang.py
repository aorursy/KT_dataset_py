#ECE 475 Freq Makeene Learning Kaggle Competition
#Jing Jiang

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score

df_train = pd.read_csv("../input/trainFeatures.csv")
df_test = pd.read_csv("../input/testFeatures.csv")
df_label = pd.read_csv("../input/trainLabels.csv")
df_train.head()

# Any results you write to the current directory are saved as output.
from sklearn.preprocessing import Imputer

x = df_train.drop(['ids'], axis=1).select_dtypes(exclude=['object']).fillna(0)
x_test = df_test.drop(['ids'], axis=1).select_dtypes(exclude=['object']).fillna(0)
y = df_label.drop(['ids'], axis=1).drop(['ATScore'], axis=1)
y_AT = df_label.drop(['ids'], axis=1).drop(['OverallScore'], axis=1)

my_imputer = Imputer()
x = my_imputer.fit_transform(x)
x_test = my_imputer.transform(x_test)
y = my_imputer.fit_transform(y)
y_AT = my_imputer.fit_transform(y_AT)
# Regression (Simple Linear Regression)
from sklearn.linear_model import LinearRegression

lireg = LinearRegression().fit(x, y)

lireg_pred = lireg.predict(x_test)
lireg_pred = lireg_pred[:,0]

my_submission3 = pd.DataFrame({'Id': df_test.ids, 'OverallScore': lireg_pred})
my_submission3.to_csv('submission_linear_JJ.csv', index=False)

# model performed on ATScore label
lireg2 = LinearRegression().fit(x, y_AT)
lireg_pred2 = lireg2.predict(x_test)

print("The Cross-val score for ATScore using Simple Linear Regression is: ")
print(cross_val_score(lireg2, x, y_AT))

# first five of OverallScore submission
my_submission3.head() 
# Regression (L2 Penalty)
from sklearn.linear_model import Ridge
rid = Ridge(alpha=1.0)
rid.fit(x, y) 

rid_predictions = rid.predict(x_test)
rid_predictions = rid_predictions[:,0]

my_submission1 = pd.DataFrame({'Id': df_test.ids, 'OverallScore': rid_predictions})
my_submission1.to_csv('submission_ridge_JJ.csv', index=False)

# model performed on ATScore label
rid2 = rid.fit(x, y_AT)
rid_pred2 = rid2.predict(x_test)

print("The Cross-val score for ATScore using Linear Regression with L2 Penalty is: ")
print(cross_val_score(rid2, x, y_AT))

# first five of OverallScore submission
my_submission1.head() 
# Regression (L1 Penalty)
from sklearn import linear_model
las = linear_model.Lasso(alpha=0.1)
las.fit(x, y)

las_predictions = las.predict(x_test)

my_submission2 = pd.DataFrame({'Id': df_test.ids, 'OverallScore': las_predictions})
my_submission2.to_csv('submission_lasso_JJ.csv', index=False)

# model performed on ATScore label
las2 = las.fit(x, y_AT)
las_pred2 = las2.predict(x_test)

print("The Cross-val score for ATScore using Linear Regression with L1 Penalty is: ")
print(cross_val_score(las2, x, y_AT))

# first five of OverallScore submission
my_submission2.head() 
# XGBoost
# from xgboost import XGBRegressor
import xgboost as xgb
from scipy import stats
from sklearn.model_selection import *
from sklearn.model_selection import GridSearchCV

xg_model = xgb.XGBRegressor()
parameters1 = {'max_depth': [4, 5],
              'min_child_weight': [1, 2],
               'learning_rate': [0.06, 0.07],
              'subsample': [0.8],
              'colsample_bytree': [0.95],
              'n_estimators': [2300, 2350],
                'gamma': [0.4],
              'res_alpha': [0.15],
              'reg_lambda': [3],
              'seed': [0]}
xgcv = GridSearchCV(xg_model, parameters1, n_jobs=5, 
                   cv=2, 
                   verbose=True)
xgcv.fit(x, y)
print(xgcv.best_params_) # print best parameters using GridSearchCV

xg_pred = xgcv.predict(x_test)
xg_pred

my_submission4 = pd.DataFrame({'Id': df_test.ids, 'OverallScore': xg_pred})
my_submission4.to_csv('submission_xgboost_JJ.csv', index=False)
import xgboost as xgb
# model performed on ATScore label
xg_model_AT = xgb.XGBRegressor(colsample_bytree=0.95, 
                         gamma=0.4, learning_rate=0.07, max_depth=4, 
                         min_child_weight=2, n_estimators=2350,reg_lambda=3, 
                         res_alpha=0.15, seed=0, subsample=0.8)
xgAT = xg_model_AT.fit(x, y_AT)
xgAT_pred = xgAT.predict(x_test)

print("The Cross-val score for ATScore using XGBoost is: ")
print(cross_val_score(xg_model_AT, x, y_AT))

# first five of OverallScore submission
my_submission4.head() 
# Elastic Net Regression (A type of Regression we did not cover in class)
from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression

enregr = ElasticNet(random_state=0)
enregr.fit(x, y)

en_pred = enregr.predict(x_test)
en_pred

my_submission5 = pd.DataFrame({'Id': df_test.ids, 'OverallScore': en_pred})
my_submission5.to_csv('submission_elastic_net_regr_JJ.csv', index=False)

# model performed on ATScore label
en2 = enregr.fit(x, y_AT)
en_pred2 = en2.predict(x_test)

print("The Cross-val score for ATScore using Elastic Net Regression is: ")
print(cross_val_score(en2, x, y_AT))

# first five of OverallScore submission
my_submission5.head() 
# Random Forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

regr = RandomForestRegressor(max_depth=2, random_state=0,
                             n_estimators=100)
regr.fit(x, y)

rf_pred = regr.predict(x_test)
rf_pred

my_submission6 = pd.DataFrame({'Id': df_test.ids, 'OverallScore': rf_pred})
my_submission6.to_csv('submission_random_forest_JJ.csv', index=False)

# model performed on ATScore label
rf2 = regr.fit(x, y_AT)
rf_pred2 = rf2.predict(x_test)
print("The Cross-val score for ATScore using Random Forest is: ")
print(cross_val_score(rf2, x, y_AT))

# first five of OverallScore submission
my_submission6.head() 
# Dimensionality Reduction using PCA (Principal Component Analysis)
from sklearn.decomposition import PCA

pca = PCA(.95) # scikit-learn chooses the minimum number of principal components such that 95% of the variance is retained.
pca = pca.fit(x)

x_pca = pca.transform(x)
x_test_pca = pca.transform(x_test)
# Performs XGBoost Regressor on data frame after Dimensionality Reduction with PCA

from xgboost import XGBRegressor

dr = XGBRegressor(colsample_bytree=0.95, 
                         gamma=0.4, learning_rate=0.07, max_depth=4, 
                         min_child_weight=2, n_estimators=2350,reg_lambda=3, 
                         res_alpha=0.15, seed=0, subsample=0.8)
# Add silent=True to avoid printing out updates with each cycle
dr.fit(x_pca, y, verbose=False)

dr_pred = dr.predict(x_test_pca)
dr_pred

my_submission7 = pd.DataFrame({'Id': df_test.ids, 'OverallScore': dr_pred})
my_submission7.to_csv('submission_dimension_reduc_JJ.csv', index=False)

# model performed on ATScore label
dr2 = dr.fit(x_pca, y_AT)
dr_pred2 = dr2.predict(x_test_pca)

print("The Cross-val score for ATScore using Dimensionality Reduction is: ")
print(cross_val_score(dr2, x_pca, y_AT))

# first five of OverallScore submission
my_submission7.head()
# Feature Selection using Univariate Feature Selection with selectKBest, which removes all but the k highest scoring features
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression

select = SelectKBest(mutual_info_regression, k=20).fit(x, y) # score_func = mutual_info_regression, k = 20
x_new = select.transform(x)
x_test_new = select.transform(x_test)
# Performs XGBoost Regressor on data frame after Feature Selection
from xgboost import XGBRegressor

fs = XGBRegressor(colsample_bytree=0.95, 
                         gamma=0.4, learning_rate=0.07, max_depth=4, 
                         min_child_weight=2, n_estimators=2350,reg_lambda=3, 
                         res_alpha=0.15, seed=0, subsample=0.8)
# Add silent=True to avoid printing out updates with each cycle
fs.fit(x_new, y, verbose=False)

fs_pred = fs.predict(x_test_new)

my_submission8 = pd.DataFrame({'Id': df_test.ids, 'OverallScore': fs_pred})
my_submission8.to_csv('submission_feature_select_JJ.csv', index=False)

# model performed on ATScore label
fs2 = fs.fit(x_new, y_AT)
fs_pred = fs2.predict(x_test_new)

print("The Cross-val score for ATScore using Feature Selection is: ")
print(cross_val_score(fs2, x_new, y_AT))

# first five of OverallScore submission
my_submission8.head() 