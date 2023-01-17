# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np  # linear algebra

import pandas as pd  #

from datetime import datetime



from scipy.stats import skew  # for some statistics

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax



from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.svm import SVR

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error, mean_absolute_error



from mlxtend.regressor import StackingCVRegressor



from xgboost import XGBRegressor

from lightgbm import LGBMRegressor
train = pd.read_csv('../input/train.csv', index_col=0)

test = pd.read_csv('../input/test.csv', index_col=0)

print("Train set size:", train.shape)

print("Test set size:", test.shape)
train = train.drop(train[(train['administracion']>1.1)].index)

train = train.drop(train[(train['administracion']<0.2) & (train['precio']>750)].index)
train["precio"] = np.log1p(train["precio"])

y = train.precio.reset_index(drop=True)

train_features = train.drop(['precio'], axis=1)

test_features = test
features = pd.concat([train_features, test_features]).reset_index(drop=True)

print(features.shape)
print(features.shape)

final_features = pd.get_dummies(features).reset_index(drop=True)

print(final_features.shape)



X = final_features.iloc[:len(y), :]

X_sub = final_features.iloc[len(X):, :]



print('X', X.shape, 'y', y.shape, 'X_sub', X_sub.shape)
kfolds = KFold(n_splits=10, shuffle=True, random_state=42)





# rmsle

def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))





# build our model scoring function

def cv_rmse(model):

    rmse = np.sqrt(-cross_val_score(model, X, y,

                                    scoring="neg_mean_squared_error",

                                    cv=kfolds))

    return rmse
alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]

alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]

e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]

e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]



ridge = make_pipeline(RobustScaler(),

                      RidgeCV(alphas=alphas_alt, cv=kfolds))



lasso = make_pipeline(RobustScaler(),

                      LassoCV(max_iter=1e7, alphas=alphas2,

                              random_state=42, cv=kfolds))



elasticnet = make_pipeline(RobustScaler(),

                           ElasticNetCV(max_iter=1e7, alphas=e_alphas,

                                        cv=kfolds, l1_ratio=e_l1ratio))



svr = make_pipeline(RobustScaler(),

                    SVR(C=20, epsilon=0.008, gamma=0.0003, ))



gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                max_depth=4, max_features='sqrt',

                                min_samples_leaf=15, min_samples_split=10,

                                loss='huber', random_state=42)



lightgbm = LGBMRegressor(objective='regression',

                         num_leaves=4,

                         learning_rate=0.01,

                         n_estimators=5000,

                         max_bin=200,

                         bagging_fraction=0.75,

                         bagging_freq=5,

                         bagging_seed=7,

                         feature_fraction=0.2,

                         feature_fraction_seed=7,

                         verbose=-1,

                         # min_data_in_leaf=2,

                         # min_sum_hessian_in_leaf=11

                         )



xgboost = XGBRegressor(learning_rate=0.01, n_estimators=3460,

                       max_depth=3, min_child_weight=0,

                       gamma=0, subsample=0.7,

                       colsample_bytree=0.7,

                       objective='reg:linear', nthread=-1,

                       scale_pos_weight=1, seed=27,

                       reg_alpha=0.00006)
stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet,

                                            gbr, xgboost, lightgbm),

                                meta_regressor=xgboost,

                                use_features_in_secondary=True)
score = cv_rmse(ridge)

print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



score = cv_rmse(lasso)

print("Lasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



score = cv_rmse(elasticnet)

print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



score = cv_rmse(svr)

print("SVR score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



score = cv_rmse(gbr)

print("GradientBoosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



score = cv_rmse(lightgbm)

print("Lightgbm score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



score = cv_rmse(xgboost)

print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )
print(datetime.now(), 'StackingCVRegressor')

stack_gen_model = stack_gen.fit(np.array(X), np.array(y))

print(datetime.now(), 'elasticnet')

elastic_model_full_data = elasticnet.fit(X, y)

print(datetime.now(), 'lasso')

lasso_model_full_data = lasso.fit(X, y)

print(datetime.now(), 'ridge')

ridge_model_full_data = ridge.fit(X, y)

print(datetime.now(), 'svr')

svr_model_full_data = svr.fit(X, y)

print(datetime.now(), 'GradientBoosting')

gbr_model_full_data = gbr.fit(X, y)

print(datetime.now(), 'xgboost')

xgb_model_full_data = xgboost.fit(X, y)

print(datetime.now(), 'lightgbm')

lgb_model_full_data = lightgbm.fit(X, y)
def blend_models_predict(X=X):

    return ((0.1* elastic_model_full_data.predict(X)) + 

            (0.1 * lasso_model_full_data.predict(X)) + 

            (0.05 * ridge_model_full_data.predict(X)) + 

            (0.1 * svr_model_full_data.predict(X)) + 

            (0.1 * gbr_model_full_data.predict(X)) + 

            (0.15 * xgb_model_full_data.predict(X)) + 

            (0.1 * lgb_model_full_data.predict(X)) + 

            (0.3 * stack_gen_model.predict(np.array(X))))
print('RMSLE score on train data:')

print(rmsle(y, blend_models_predict(X)))

print('MSE score on train data:')

print(mean_squared_error(y, blend_models_predict(X)))

print('MAE score on train data:')

print(mean_absolute_error(np.expm1(y), np.floor(np.expm1(blend_models_predict(X)))))
submission = pd.DataFrame()

submission['Precio'] = np.floor(np.expm1(blend_models_predict(X_sub)))

submission.index += 1

submission.index.names = ['Apartamento']

submission.to_csv('submission.csv')