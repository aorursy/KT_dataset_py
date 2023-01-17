import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import GridSearchCV, KFold
train = pd.read_csv('../input/housees/train.csv')
test = pd.read_csv('../input/housees/test.csv')
train.head()
train_target = train['SalePrice']
tr_tst = pd.concat([train.drop('SalePrice',axis=1),test]).reset_index(drop=True)
print(tr_tst.info())
tr_tst['GarageYrBlt'] = tr_tst['GarageYrBlt'].fillna(0)

for col in ['Exterior1st', 'Exterior2nd', 'SaleType']:
  tr_tst[col] = tr_tst[col].fillna(tr_tst[col].mode()[0])
 
for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 
            'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']:
    tr_tst[col] = tr_tst[col].fillna('None')

tr_tst['LotFrontage'] = tr_tst.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
numeric_cols = []
for col in tr_tst.columns:
    if tr_tst[col].dtype in ['int64','float64']:
        numeric_cols.append(col)
tr_tst.update(tr_tst[numeric_cols].fillna(0))
object_col = []
for col in tr_tst.columns:
    if tr_tst[col].dtype == object:
        object_col.append(col)
tr_tst.update(tr_tst[object_col].fillna('None'))

tr_tst['MSZoning'] = tr_tst.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(tr_tst.isna().sum(axis=0))
tr_tst_num= tr_tst.select_dtypes(include=['float64','int64']).columns  
tr_tst_cat = tr_tst.select_dtypes(exclude=['float64','int64']) 
tr_tst_cat_dummy= pd.get_dummies(tr_tst_cat)
tr_tst=pd.concat([tr_tst,tr_tst_cat_dummy],axis=1) 
tr_tst= tr_tst.drop(tr_tst_cat.columns,axis=1) 
X = tr_tst[:len(train)] 
test = tr_tst[len(train):]
kfold= KFold(n_splits=10,shuffle=True)
X_train, X_test, y_train, y_test = train_test_split(X, train_target, test_size=0.2)
from sklearn.linear_model import Lasso, Ridge
r=Ridge()
params= {'alpha': [5,8,10,10.1,10.2,10.3,10.4,10.5,11,12,15]}

grid_ridge = GridSearchCV(r, param_grid=params,cv=kfold,scoring='neg_mean_squared_error')
grid_ridge.fit(X_train, y_train)

alpha = grid_ridge.best_params_
ridge_score = grid_ridge.best_score_
print("The best alpha:",alpha['alpha'])
R = Ridge(alpha=12)

R.fit(X_train, y_train)

train_pred = R.predict(X_train)
test_pred = R.predict(X_test)

for i in range(len(test_pred)):
  if test_pred[i] <0:
    test_pred[i]=0

print('RMSE train = ', mean_squared_log_error(y_train, train_pred))
print('RMSE test = ', mean_squared_log_error(y_test, test_pred))
r=Lasso(max_iter=10000)
params= {'alpha': np.linspace(5, 35, 15)}

grid_ridge = GridSearchCV(r, param_grid=params,cv=kfold,scoring='neg_mean_squared_error')
grid_ridge.fit(X_train, y_train)

alpha = grid_ridge.best_params_
ridge_score = grid_ridge.best_score_
print("The best alpha:",alpha['alpha'])
L = Lasso(alpha=35)

L.fit(X_train, y_train)

train_pred = L.predict(X_train)
test_pred = L.predict(X_test)

for i in range(len(test_pred)):
  if test_pred[i] <0:
    test_pred[i]=0
    
print('RMSE train = ', mean_squared_log_error(y_train, train_pred))
print('RMSE test = ', mean_squared_log_error(y_test, test_pred))
from xgboost import XGBRegressor
xgb = XGBRegressor(learning_rate=0.01,n_estimators=3460, max_depth=3, min_child_weight=0, gamma=0, subsample=0.7,
                                     colsample_bytree=0.7, objective='reg:squarederror', nthread=-1,
                                     scale_pos_weight=1, seed=27, reg_alpha=0.00006)

xgb.fit(X_train, y_train)

train_pred = xgb.predict(X_train)
test_pred = xgb.predict(X_test)

for i in range(len(test_pred)):
  if test_pred[i] <0:
    test_pred[i]=0
    
print('RMSE train = ', mean_squared_log_error(y_train, train_pred))
print('RMSE test = ', mean_squared_log_error(y_test, test_pred))
from lightgbm import LGBMRegressor
lgbm =  LGBMRegressor(objective='regression', num_leaves=4,learning_rate=0.01, n_estimators=6000,
                                       max_bin=200, bagging_fraction=0.75,bagging_freq=5, bagging_seed=7,
                                       feature_fraction=0.2,feature_fraction_seed=7,verbose=-1)

lgbm.fit(X_train, y_train)

train_pred = lgbm.predict(X_train)
test_pred = lgbm.predict(X_test)

for i in range(len(test_pred)):
  if test_pred[i] <0:
    test_pred[i]=0
    
print('RMSE train = ', mean_squared_log_error(y_train, train_pred))
print('RMSE test = ', mean_squared_log_error(y_test, test_pred))
from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor(n_estimators=6000,learning_rate=0.01,max_depth=3,\
                              min_samples_leaf=15,max_features='sqrt',min_samples_split=10,loss='huber',\
                              random_state=42)

gbr.fit(X_train, y_train)

train_pred = gbr.predict(X_train)
test_pred = gbr.predict(X_test)

for i in range(len(test_pred)):
  if test_pred[i] <0:
    test_pred[i]=0
    
print('RMSE train = ', mean_squared_log_error(y_train, train_pred))
print('RMSE test = ', mean_squared_log_error(y_test, test_pred))
ans_train = pd.concat([pd.DataFrame(xgb.predict(X_train)), 
                       pd.DataFrame(L.predict(X_train)), 
                       pd.DataFrame(R.predict(X_train)),
                       pd.DataFrame(lgbm.predict(X_train)),
                       pd.DataFrame(gbr.predict(X_train))],
                       axis=1)
ans_test = pd.concat([pd.DataFrame(xgb.predict(X_test)), 
                      pd.DataFrame(L.predict(X_test)), 
                      pd.DataFrame(R.predict(X_test)),
                      pd.DataFrame(lgbm.predict(X_test)),
                      pd.DataFrame(gbr.predict(X_test))], 
                     axis=1)
ans_train.columns = [0, 1, 2, 3, 4]
ans_test.columns = [0, 1, 2, 3, 4]
xgb_stack = XGBRegressor(learning_rate=0.01,n_estimators=3460, max_depth=3, min_child_weight=0, gamma=0, subsample=0.7,
                                     colsample_bytree=0.7, objective='reg:squarederror', nthread=-1,
                                     scale_pos_weight=1, seed=27, reg_alpha=0.00006)

xgb_stack.fit(ans_train, y_train)

train_pred = xgb_stack.predict(ans_train)
test_pred = xgb_stack.predict(ans_test)

for i in range(len(test_pred)):
  if test_pred[i] <0:
    test_pred[i]=0
    
print('RMSE train = ', mean_squared_log_error(y_train, train_pred))
print('RMSE test = ', mean_squared_log_error(y_test, test_pred))
ans_test = pd.concat([pd.DataFrame(xgb.predict(test)), 
                      pd.DataFrame(L.predict(test)), 
                      pd.DataFrame(R.predict(test)),
                      pd.DataFrame(lgbm.predict(test)),
                      pd.DataFrame(gbr.predict(test))], 
                     axis=1)
ans_test.columns = [0, 1, 2, 3, 4]
test_target = (0.2*xgb.predict(test) + \
              0.15*L.predict(test) + \
              0.05*R.predict(test)  + \
              0.2*lgbm.predict(test) + \
              0.2*xgb_stack.predict(ans_test) + \
              0.2*gbr.predict(test))
submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
#test_target = R.predict(test)
submission['SalePrice'] = test_target
submission.to_csv("submission_prediction.csv", index=False)