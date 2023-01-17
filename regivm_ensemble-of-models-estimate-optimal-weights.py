import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.preprocessing import RobustScaler
from sklearn import ensemble
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score
from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from scipy.optimize import minimize

from datetime import datetime

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_score, train_test_split, RandomizedSearchCV
from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import os
X = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')
comp = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')

print("Train set size:", X.shape)
print("Test set size:", comp.shape)
print('START data processing',   )
from scipy.stats import norm
sns.distplot(X["SalePrice"],fit=norm)
mu,sigma= norm.fit(X['SalePrice'])
print("mu {}, sigma {}".format(mu,sigma))
########## REMOVING SKEWEENESS ###########

X['SalePrice']=np.log1p(X['SalePrice'])
sns.distplot(X["SalePrice"],fit=norm)
mu,sigma= norm.fit(X['SalePrice'])
print("mu {}, sigma {}".format(mu,sigma))
X_ID = X['Id']
comp_ID = comp['Id']
# Now drop the  'Id' colum since it's unnecessary for  the prediction process.
X.drop(['Id'], axis=1, inplace=True)
comp.drop(['Id'], axis=1, inplace=True)

# Deleting outliers
X = X[X.GrLivArea < 4500]
X.reset_index(drop=True, inplace=True)

# We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
X["SalePrice"] = np.log1p(X["SalePrice"])
y = X.SalePrice.reset_index(drop=True)
X = X.drop(['SalePrice'], axis=1)
X_comp = comp
X_all = pd.concat([X, X_comp]).reset_index(drop=True)
print(X_all.shape)
# Some of the non-numeric predictors are stored as numbers; we convert them into strings 
X_all['MSSubClass'] = X_all['MSSubClass'].apply(str)
X_all['YrSold'] = X_all['YrSold'].astype(str)
X_all['MoSold'] = X_all['MoSold'].astype(str)

X_all['Functional'] = X_all['Functional'].fillna('Typ')
X_all['Electrical'] = X_all['Electrical'].fillna("SBrkr")
X_all['KitchenQual'] = X_all['KitchenQual'].fillna("TA")
X_all['Exterior1st'] = X_all['Exterior1st'].fillna(X_all['Exterior1st'].mode()[0])
X_all['Exterior2nd'] = X_all['Exterior2nd'].fillna(X_all['Exterior2nd'].mode()[0])
X_all['SaleType'] = X_all['SaleType'].fillna(X_all['SaleType'].mode()[0])

X_all["PoolQC"] = X_all["PoolQC"].fillna("None")

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    X_all[col] = X_all[col].fillna(0)
for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    X_all[col] = X_all[col].fillna('None')
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    X_all[col] = X_all[col].fillna('None')

X_all['MSZoning'] = X_all.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

objects = list(X_all.select_dtypes(include='object').columns)

X_all[objects]=X_all[objects].fillna('None')

X_all['LotFrontage'] = X_all.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

# Filling in the rest of the NA's
numerics = list(X_all.select_dtypes(exclude='object').columns)
X_all[numerics] = X_all[numerics].fillna(0)
numerics2 = list(X_all.select_dtypes(exclude='object').columns)

skew_features = X_all[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)

high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index

for i in skew_index:
    X_all[i] = boxcox1p(X_all[i], boxcox_normmax(X_all[i] + 1))

X_all = X_all.drop(['Utilities', 'Street', 'PoolQC', ], axis=1)

X_all['YrBltAndRemod'] = X_all['YearBuilt'] + X_all['YearRemodAdd']
X_all['TotalSF'] = X_all['TotalBsmtSF'] + X_all['1stFlrSF'] + X_all['2ndFlrSF']

X_all['Total_sqr_footage'] = (X_all['BsmtFinSF1'] + X_all['BsmtFinSF2'] +
                                 X_all['1stFlrSF'] + X_all['2ndFlrSF'])

X_all['Total_Bathrooms'] = (X_all['FullBath'] + (0.5 * X_all['HalfBath']) +
                               X_all['BsmtFullBath'] + (0.5 * X_all['BsmtHalfBath']))

X_all['Total_porch_sf'] = (X_all['OpenPorchSF'] + X_all['3SsnPorch'] +
                    X_all['EnclosedPorch'] + X_all['ScreenPorch'] + X_all['WoodDeckSF'])

# simplified features
X_all['haspool'] = X_all['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
X_all['has2ndfloor'] = X_all['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
X_all['hasgarage'] = X_all['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
X_all['hasbsmt'] = X_all['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
X_all['hasfireplace'] = X_all['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

print(X_all.shape)
final_X_all = pd.get_dummies(X_all).reset_index(drop=True)
print(final_X_all.shape)

X = final_X_all.iloc[:len(y), :]
X_comp = final_X_all.iloc[len(X):, :]

print('X', X.shape, 'y', y.shape, 'X_comp', X_comp.shape)

outliers = [30, 88, 462, 631, 1322]
X = X.drop(X.index[outliers])
y = y.drop(y.index[outliers])

overfit = []
for i in X.columns:
    counts = X[i].value_counts()
    zeros = counts.iloc[0]
    if zeros / len(X) * 100 > 99.94:
        overfit.append(i)

overfit = list(overfit)
overfit.append('MSZoning_C (all)')

X = X.drop(overfit, axis=1).copy()
X_comp = X_comp.drop(overfit, axis=1).copy()

print('X', X.shape, 'y', y.shape, 'X_comp', X_comp.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
niter=20
ncv = 3
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
xgb = XGBRegressor(objective='reg:squarederror')

params = {'n_estimators': sp_randint(100, 6000),
         'max_depth' :sp_randint(1, 50),
         'sub_sample': sp_uniform(0, 1),
         'colsample_bytree': sp_uniform(0, 1),
         'reg_alpha': sp_uniform(0,1)}

rsearch = RandomizedSearchCV(xgb, param_distributions=params, n_iter=niter, cv=ncv, 
                             scoring='neg_root_mean_squared_error', n_jobs=-1)
#rsearch.fit(X, y)
#xgb_params = rsearch.best_params_
xgb_params = {'colsample_bytree': 0.529945192578405,
 'max_depth': 1,
 'n_estimators': 2377,
 'reg_alpha': 0.22958335326689505,
 'sub_sample': 0.824538323669472}
lgbm = LGBMRegressor(objective='regression')

params = {'n_estimators': sp_randint(100, 6000),
         'max_depth' :sp_randint(1, 150),
          'num_leaves': sp_randint(10, 2500),
         'bagging_fraction': sp_uniform(0, 1),
         'feature_fraction': sp_uniform(0, 1),
         'learning_rate': sp_uniform(0,1)}

rsearch = RandomizedSearchCV(lgbm, param_distributions=params, n_iter=niter, cv=ncv, 
                             scoring='neg_root_mean_squared_error')
#rsearch.fit(X, y)
#lgbm_params = rsearch.best_params_
lgbm_params = {'bagging_fraction': 0.3826025415942872,
 'feature_fraction': 0.4152600001975988,
 'learning_rate': 0.03491619499427312,
 'max_depth': 2,
 'n_estimators': 3496,
 'num_leaves': 1323}
gbr = GradientBoostingRegressor()

params = {'n_estimators': sp_randint(100, 6000),
         'max_depth' :sp_randint(1, 150),
          'max_features': sp_randint(10, 330),
         'min_samples_split': sp_randint(2, 100),
         'min_samples_leaf': sp_randint(1, 100),
         'learning_rate': sp_uniform(0,1)}

rsearch = RandomizedSearchCV(gbr, param_distributions=params, n_iter=niter, cv=ncv, 
                             scoring='neg_root_mean_squared_error', n_jobs=-1)
#rsearch.fit(X, y)
#gbr_params = rsearch.best_params_
gbr_params = {'learning_rate': 0.3762972471885566,
 'max_depth': 63,
 'max_features': 121,
 'min_samples_leaf': 19,
 'min_samples_split': 73,
 'n_estimators': 5546}
rfr = RandomForestRegressor()

params = {'n_estimators': sp_randint(100, 6000),
         'max_depth' :sp_randint(1, 16),
         'min_samples_leaf': sp_randint(1, 100), 
         'min_samples_split': sp_randint(2, 100), 
         'max_features': sp_randint(10,330)}

rsearch = RandomizedSearchCV(rfr, param_distributions=params, n_iter=niter, cv=ncv, 
                             scoring='neg_root_mean_squared_error', n_jobs=-1)
#rsearch.fit(X, y)
#rfr_params = rsearch.best_params_
rfr_params = {'max_depth': 5,
 'max_features': 185,
 'min_samples_leaf': 1,
 'min_samples_split': 22,
 'n_estimators': 959}
gbr = ensemble.GradientBoostingRegressor(loss='ls', **gbr_params)

xgb = XGBRegressor(objective='reg:squarederror', **xgb_params)

lgbm = LGBMRegressor(objective='regression', **lgbm_params)

rfr = RandomForestRegressor(**rfr_params)

cb = CatBoostRegressor(loss_function='RMSE', logging_level='Silent')
def mean_cross_val(model, X, y):
    score = -cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
    mean = score.mean()
    return mean

clfs = []

gbr.fit(X_train, y_train)   
preds = gbr.predict(X_test) 
rmse_gbr = np.sqrt(mean_squared_error(y_test, preds))
score_gbr = gbr.score(X_test, y_test)
cv_gbr = mean_cross_val(gbr, X, y)
print('gbr:  ', cv_gbr)
clfs.append(gbr)

xgb.fit(X_train, y_train)   
preds = xgb.predict(X_test) 
rmse_xgb = np.sqrt(mean_squared_error(y_test, preds))
score_xgb = xgb.score(X_test, y_test)
cv_xgb = mean_cross_val(xgb, X, y)
print('xgb:  ', cv_xgb)
clfs.append(xgb)


lgbm.fit(X_train, y_train)   
preds = lgbm.predict(X_test) 
rmse_lgbm = np.sqrt(mean_squared_error(y_test, preds))
score_lgbm = lgbm.score(X_test, y_test)
cv_lgbm = mean_cross_val(lgbm, X, y)
print('lgbm:  ', cv_lgbm)
clfs.append(lgbm)


rfr.fit(X_train, y_train)   
preds = rfr.predict(X_test) 
rmse_rfr = np.sqrt(mean_squared_error(y_test, preds))
score_rfr = rfr.score(X_test, y_test)
cv_rfr = mean_cross_val(rfr, X, y)
print('rfr:  ', cv_rfr)
clfs.append(rfr)


cb.fit(X_train, y_train)   
preds = cb.predict(X_test) 
rmse_cb = np.sqrt(mean_squared_error(y_test, preds))
score_cb = rfr.score(X_test, y_test)
cv_cb = mean_cross_val(rfr, X, y)
print('cb:  ', cv_cb)
clfs.append(cb)
model_performances = pd.DataFrame({
    "Model" : ["GradBRegr", "XGBoost", "LGBM", "RandomForest"],
    "CV(5)" : [str(cv_gbr)[0:5], str(cv_xgb)[0:5], str(cv_lgbm)[0:5], str(cv_rfr)[0:5]],
    "RMSE" : [str(rmse_gbr)[0:5], str(rmse_xgb)[0:5], str(rmse_lgbm)[0:5], str(rmse_rfr)[0:5]],
    "Score" : [str(score_gbr)[0:5], str(score_xgb)[0:5], str(score_lgbm)[0:5], str(score_rfr)[0:5]]
})

print("Sorted by Score:")
print(model_performances.sort_values(by="Score", ascending=False))

def blend_models_predict(X1, a, b, c, d, e):
    return (a*gbr.predict(X1) + b*xgb.predict(X1) + c*lgbm.predict(X1) + d*rfr.predict(X1) + e*cb.predict(X1))
## Optimization

predictions = []
for clf in clfs:
    predictions.append(clf.predict(X_test))

def mse_func(weights):
    #scipy minimize will pass the weights as a numpy array
    final_prediction = 0
    for weight, prediction in zip(weights, predictions):
            final_prediction += weight*prediction
    #return np.mean((y_test-final_prediction)**2)
    return np.sqrt(mean_squared_error(y_test, final_prediction))
    
starting_values = [0]*len(predictions)

cons = ({'type':'ineq','fun':lambda w: 1-sum(w)})
#our weights are bound between 0 and 1
bounds = [(0,1)]*len(predictions)

res = minimize(mse_func, starting_values, method='SLSQP', bounds=bounds, constraints=cons)

print('Ensamble Score: {best_score}'.format(best_score=res['fun']))
print('Best Weights: {weights}'.format(weights=res['x']))
yy = 'abcde'
dd={yy[i]: k for i,k in enumerate(res['x'])}
print(dd)
def blend_models_predict(X1, **dd):
    return (dd['a']*gbr.predict(X1) + dd['b']*xgb.predict(X1) + 
            dd['c']*lgbm.predict(X1) + dd['d']*rfr.predict(X1) + dd['e']*rfr.predict(X1))
subm = np.exp(blend_models_predict(X_comp, **dd))
submission = pd.DataFrame({'Id': comp_ID, 'SalePrice': subm})
submission.to_csv("submission36.csv", index=False)
print('complete')