import pandas as pd

import numpy as np

import numpy.random as random

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import mean_squared_error, r2_score

import xgboost as xgb

from xgboost.sklearn import XGBRegressor

from sklearn import preprocessing

import time

import os

import datetime
DIRNAME = None  # name of the result folder

# STD_IN = True   # standartize independent vars
df_test = pd.read_csv('../input/test.csv')

df_train = pd.read_csv('../input/train.csv')

ytrain = df_train.pop('SalePrice')

df_all = pd.concat((df_train, df_test), ignore_index=True)

y_idx = df_all.pop('Id')
df_all = pd.get_dummies(df_all)

df_all = df_all.fillna(df_all.mean())

train_sz = ytrain.shape[0]

xtrain = df_all.iloc[:train_sz].copy()

xtest = df_all.iloc[train_sz:].copy()

del df_all

del df_train
# Standartization of independent vars

scaler = preprocessing.StandardScaler(with_mean=True, with_std=True).fit(xtrain)

xtrain_ss = pd.DataFrame(scaler.transform(xtrain), columns=xtrain.columns)

xtest_ss = pd.DataFrame(scaler.transform(xtest), columns=xtest.columns)

yss_vector = preprocessing.StandardScaler(with_mean=True, with_std=True).fit_transform(

    ytrain.astype('float').values.reshape(-1, 1))

ytrain_ss = pd.Series(yss_vector.flatten(), name='SalePrice')
def mkdir():

    dirname = datetime.datetime.now().strftime('subscr-%d%m.%H:%M:%S')

    try:

        os.mkdir(dirname)

    except FileExistsError:

        pass

    return dirname



def save_res(y_pred, fname):

    ''' Save predicted values as csv file '''

    global DIRNAME

    DIRNAME = DIRNAME or mkdir()

    res = pd.DataFrame({'Id': df_test.Id, 'SalePrice': y_pred}, index=None)

    if not fname.endswith('.csv'):

        fname += '.csv'

    fpath = os.path.join(DIRNAME, fname)

    res.to_csv(fpath, header=True, index=False)

    print("The result has been saved as", fpath)
# initial parameters

REGR_PARAMS = {

    'learning_rate': 0.1, 'subsample': 0.8,

    'n_estimators': 100, 

    'seed':0, 

    'colsample_bytree': 0.8, 

    'objective': 'reg:linear',

    'booster': 'gbtree',

    'max_depth': 3, 'min_child_weight': 1

}



# Each row is one optimization step

OPT_STEPS = [

    {'n_estimators': range(300, 1100, 100)},

    {'booster': ['gbtree', 'gblinear', 'dart']},

    {'max_depth': range(2, 8), 'min_child_weight': [1,3,5]},

    {'learning_rate': [0.01, 0.05, 0.1, 0.5], 'subsample': [0.7,0.8,0.9]}

]



SCORING = 'r2'
regr_params = REGR_PARAMS.copy()

opt_gbm = None

for idx, cv_params in enumerate(OPT_STEPS):

    time0 = time.monotonic()

    print("{}: Optimize by {}".format(idx+1, list(cv_params.keys())))

    opt_gbm = GridSearchCV(xgb.XGBRegressor(**regr_params), 

                           cv_params, scoring = SCORING, cv = 5, n_jobs = -1)

    opt_gbm.fit(xtrain, ytrain)

    dtime = time.monotonic() - time0

    print("{}: {:.3f} sec. best score: {:.6f}, best params: {}".format(

        idx+1, dtime, opt_gbm.best_score_, opt_gbm.best_params_))

    # replace the tested values to the best found

    regr_params.update(opt_gbm.best_params_)

gbm_param = opt_gbm.best_estimator_.get_xgb_params()

print("Optimized parameters: {}".format(gbm_param))
# xgb.cv() allows to restrict count of steps using early stop

# set n_estimators?

xgtrain = xgb.DMatrix(xtrain, label=ytrain)

cv_res = xgb.cv(params=gbm_param, dtrain=xgtrain, 

                num_boost_round=gbm_param['n_estimators'], 

                nfold=5, 

                metrics='rmse', 

                early_stopping_rounds=50,

                show_stdv=True)

gbm_param['n_estimators'] = cv_res.shape[0]

gbm_param
# Get the trained booster. 

# Actually there is the alternative to use the best model 'opt_gbm.best_estimator.

# It includes the same booster. The only difference that booster accepts DMatrix, 

# when model accepts pd.DataFrame.

fin_booster = xgb.train(gbm_param, xgtrain, num_boost_round=gbm_param['n_estimators'])
fig = plt.figure(figsize=(12, 8))

ax = fig.gca()

xgb.plot_importance(fin_booster, ax=ax, max_num_features=20)
feat_imp = pd.Series(fin_booster.get_fscore()).sort_values(ascending=False)

feat_imp.head(30)
xgtrain = xgb.DMatrix(xtrain)

ytrain_pred = fin_booster.predict(xgb.DMatrix(xtrain))



dd = pd.DataFrame({'orig': ytrain, 'predict': ytrain_pred})

dd.plot(x='orig', y='predict', kind='scatter')

dd.shape
ytrain_pred.shape
xgtest = xgb.DMatrix(xtest)

y_predict = fin_booster.predict(xgtest)
save_res(y_predict, 'xgboost')
RIDGE_ALPHAS = np.logspace(-2, np.log10(800), 200)
from sklearn.linear_model import RidgeCV, Ridge



#Using mean squared error as 'cv'

rcv_ns = RidgeCV(alphas=RIDGE_ALPHAS, scoring="neg_mean_squared_error", store_cv_values=True)

r1 = rcv_ns.fit(xtrain, ytrain).score(xtrain, ytrain)

alpha_ns = rcv_ns.alpha_

print("score: {:.6f}, alpha: {:.6f}".format(r1, alpha_ns))

print('scoring function from param:', rcv_ns.get_params()['scoring'])
# Default cv function, 'x' is standartized

rcv_ss = RidgeCV(alphas=[val for val in range(500, 900, 10)], cv=5)

r1 = rcv_ss.fit(xtrain_ss, ytrain).score(xtrain_ss, ytrain)

alpha_ss = rcv_ss.alpha_

print("score: {:.6f}, alpha: {:.6f}".format(r1, alpha_ss))

print('scoring function from param:', rcv_ss.get_params()['scoring'])
# Ridge has been used for non-normalized data because of better alpha
rr = Ridge(alpha=alpha_ns)

rr.fit(xtrain, ytrain)

print('Ridge score: {:.6f}'.format(rr.score(xtrain, ytrain)))

y_pred = rr.predict(xtest)

save_res(y_pred, 'ridge')
ytrain_pred = rr.predict(xtrain)

dd = pd.DataFrame({'orig': ytrain, 'predict': ytrain_pred})

dd.plot(x='orig', y='predict', kind='scatter')
import statsmodels.api as sm

fig = sm.qqplot((ytrain_pred-ytrain).values, line='q')

plt.grid(True)
from sklearn.linear_model import Lasso, LassoCV
# Both X and y are not scaled

lcv = LassoCV(alphas=None, cv=5, copy_X=True)

lcv.fit(xtrain, ytrain)

r2_nn = lcv.score(xtrain, ytrain)

alpha_nn = lcv.alpha_

print("r2: {:.6f}, alpha: {:.3f}".format(r2_nn, alpha_nn))
# Both X and y are scaled

lcv = LassoCV(alphas=None, cv=5, copy_X=True)

lcv.fit(xtrain_ss, ytrain_ss)

r2_ss = lcv.score(xtrain_ss, ytrain_ss)

alpha_ss = lcv.alpha_

print("r2: {:.6f}, alpha: {:.3f}".format(r2_ss, alpha_ss))
# X is scaled, but 'y' isn't

lcv = LassoCV(alphas=None, cv=5, copy_X=True)

lcv.fit(xtrain_ss, ytrain)

r2_sn = lcv.score(xtrain_ss, ytrain)

alpha_sn = lcv.alpha_

print("r2: {:.6f}, alpha: {:.3f}".format(r2_sn, alpha_sn))
# Variable importance
# Three tests above show that scaling of X is sufficient for performance,

# but scaling of 'y' gave the same score but different 'alpha'.

# As the result, alpha_sn has been selected.

# Another words, the scaled version of X and unscaled of 'y'

model = Lasso(alpha=alpha_sn)

model.fit(xtrain_ss, ytrain)

coeff = pd.Series(model.coef_, index=xtrain.columns)

sorted_coeff = coeff.apply(np.abs).sort_values(ascending=False)

print("The first 30 most important variables")

mi_coeff = coeff.loc[sorted_coeff.index]

print(mi_coeff.iloc[:30])
# How many coefficients are zero?

coeff_cnt = sorted_coeff.count()

coeff_z = sorted_coeff[sorted_coeff == 0].count()

print("{} are zero out of {}".format(coeff_z, coeff_cnt))
# Show that r2 is the same as before without columns with zero coefficients

zero_cols = sorted_coeff[sorted_coeff == 0].index

X = xtrain_ss.drop(zero_cols, axis=1)

r2 = model.fit(X, ytrain).score(X, ytrain)

print("r2: {:.6f}, alpha: {:.3f}, r2-r2_sn: {:.6f}".format(r2, lcv.alpha_, r2-r2_sn))
y_pred = model.predict(xtest_ss.drop(zero_cols, axis=1))

save_res(y_pred, 'lasso')
ytrain_pred = model.predict(xtrain.drop(zero_cols, axis=1))

dd = pd.DataFrame({'orig': ytrain, 'predict': ytrain_pred})

dd.plot(x='orig', y='predict', kind='scatter')
fig = sm.qqplot((ytrain_pred-ytrain).values, line='q')

plt.grid(True)
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import AdaBoostRegressor
# X is scaled, 'y' isn't

rng = np.random.RandomState(1)  # generator

regr_sn = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),

                          n_estimators=300, random_state=rng)

r2 = regr_sn.fit(xtrain_ss, ytrain).score(xtrain_ss, ytrain)

print(regr_sn)

print('r2:', r2)
# Both X and 'y' aren't scaled

rng = np.random.RandomState(1)  # generator

regr_nn = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),

                          n_estimators=300, random_state=rng)



r2 = regr_nn.fit(xtrain, ytrain).score(xtrain, ytrain)

print(regr_nn)

print('r2:', r2)
# Both X and 'y' are scaled

rng = np.random.RandomState(1)  # generator

regr_ss = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),

                          n_estimators=300, random_state=rng)



r2 = regr_ss.fit(xtrain_ss, ytrain_ss).score(xtrain_ss, ytrain_ss)

print(regr_ss)

print('r2:', r2)
# Results are too close. The case when X and 'y' are not scaled is selected

y_pred = regr_nn.predict(xtest)

save_res(y_pred, 'adaboost')
ytrain_pred = regr_nn.predict(xtrain)

dd = pd.DataFrame({'orig': ytrain, 'predict': ytrain_pred})

dd.plot(x='orig', y='predict', kind='scatter')
fig = sm.qqplot((ytrain_pred-ytrain).values, line='q')

plt.grid(True)