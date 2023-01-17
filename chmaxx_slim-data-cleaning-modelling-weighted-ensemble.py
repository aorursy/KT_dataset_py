%matplotlib inline

%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt



plt.style.use('ggplot')

import matplotlib.cm as cm

import seaborn as sns



import pandas as pd

import pandas_profiling

import numpy as np

from numpy import percentile

from scipy import stats

from scipy.stats import skew

from scipy.special import boxcox1p



import os, sys

import calendar



from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.dummy import DummyRegressor

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.model_selection import cross_val_score, cross_validate

from sklearn.model_selection import GridSearchCV, KFold

from sklearn.decomposition import PCA

from sklearn.metrics import mean_squared_log_error, make_scorer

from sklearn.metrics.scorer import neg_mean_squared_error_scorer



from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge, RidgeCV

from sklearn.svm import SVR

from sklearn.kernel_ridge import KernelRidge

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor



import xgboost as xgb

import lightgbm as lgb





import warnings

warnings.filterwarnings('ignore')



plt.rc('font', size=18)        

plt.rc('axes', titlesize=22)      

plt.rc('axes', labelsize=18)      

plt.rc('xtick', labelsize=12)     

plt.rc('ytick', labelsize=12)     

plt.rc('legend', fontsize=12)   



plt.rcParams['font.sans-serif'] = ['Verdana']



pd.options.mode.chained_assignment = None

pd.options.display.max_seq_items = 500

pd.options.display.max_rows = 500

pd.set_option('display.float_format', lambda x: '%.5f' % x)



BASE_PATH = "/kaggle/input/house-prices-advanced-regression-techniques/"
df = pd.read_csv(f"{BASE_PATH}train.csv")

df_test = pd.read_csv(f"{BASE_PATH}test.csv")



# concat all samples to one dataframe for cleaning

# need to be careful not to leak data from test to training set! 

# e.g. by filling missing data with mean of *all* samples rather than training samples only

feat = pd.concat([df, df_test]).reset_index(drop=True).copy()
# fix missing values in features



# Alley: NA means no alley acces so we fill with string «None»

feat.Alley = feat.Alley.fillna("None")



# BsmtQual et al – NA for features means "no basement", filling with string "None"

bsmt_cols = ['BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual']

for col in bsmt_cols:

    feat[col] = feat[col].fillna("None")



# Basement sizes: NaN likely means 0, can be set to int

for col in ['BsmtFullBath', 'BsmtHalfBath', 'BsmtUnfSF', 'BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF']:

    feat[col] = feat[col].fillna(0).astype(int)

    

# Electrical: NA likely means unknown, filling with most frequent value SBrkr

feat.Electrical = feat.Electrical.fillna("SBrkr")



# Exterior1st: NA likely means unknown, filling with most frequent value VinylSd

feat.Exterior1st = feat.Exterior1st.fillna("VinylSd")



# Exterior2nd: NA likely means no 2nd material, filling with «None»

feat.Exterior2nd = feat.Exterior2nd.fillna("None")



# Fence: NA means «No Fence» filling with «None»

feat.Fence = feat.Fence.fillna("None")



# FireplaceQu: NA means «No Fireplace» filling with «None»

feat.FireplaceQu = feat.FireplaceQu.fillna("None")



# Functional: NA means «typical» filling with «Typ»

feat.Functional = feat.Functional.fillna("Typ")



# GarageType et al – NA means "no garage", filling with string "None"

grg_cols = ['GarageCond', 'GarageFinish', 'GarageQual', 'GarageType']

for col in grg_cols:

    feat[col] = feat[col].fillna("None")



# Garage sizes: NaN means «no garage» == 0, unsure if build year should be 0?

for col in ['GarageArea', 'GarageCars', 'GarageYrBlt']:

    feat[col] = feat[col].fillna(0).astype(int)



# fix one outlier GarageYrBlt == 2207

to_fix = feat[feat.GarageYrBlt == 2207].index

feat.loc[to_fix, "GarageYrBlt"] = int(feat.GarageYrBlt.mean())

    

# KitchenQual: filling NaNs with most frequent value «Typical/Average» («TA»)

feat.KitchenQual = feat.KitchenQual.fillna("TA")



# LotFrontage can be set to integer, filling missing values with 0

feat.LotFrontage = feat.LotFrontage.fillna(0).astype(int)



# MSZoning filling NaNs with most frequent value «RL» (residental low density)

feat.MSZoning = feat.MSZoning.fillna("RL")



# MSSubClass is encoded numerical but actually categorical

feat.MSSubClass = feat.MSSubClass.astype(str)



# Masonry: NA very likely means no masonry so we fill with string «None» or 0 for size

feat.MasVnrType = feat.MasVnrType.fillna("None")

feat.MasVnrArea = feat.MasVnrArea.fillna(0).astype(int)



# MiscFeature means likely no feature, filling with None

feat.MiscFeature = feat.MiscFeature.fillna("None")



# PoolQC means likely no pool, filling with None

feat.PoolQC = feat.PoolQC.fillna("None")



# SaleType: NaNs likely mean unknown, filling with most frequent value «WD»

feat.SaleType = feat.SaleType.fillna("WD")



# Utilities: NaNs likely mean unknown, filling with most frequent value «AllPub»

feat.Utilities = feat.Utilities.fillna("AllPub")
# label encode ordinal features where there is order in categories

# unfortunately can't use LabelEncoder or pd.factorize() since strings do not contain order of values



feat = feat.replace({  "Alley":        {"None" : 0, "Grvl" : 1, "Pave" : 2},

                       "BsmtCond":     {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, 

                                        "Gd" : 4, "Ex" : 5},

                       "BsmtExposure": {"None" : 0, "No" : 2, "Mn" : 2, "Av": 3, 

                                        "Gd" : 4},

                       "BsmtFinType1": {"None" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, 

                                        "BLQ" : 4, "ALQ" : 5, "GLQ" : 6},

                       "BsmtFinType2": {"None" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, 

                                        "BLQ" : 4, 

                                         "ALQ" : 5, "GLQ" : 6},

                       "BsmtQual":     {"None" : 0, "Po" : 1, "Fa" : 2, "TA": 3, 

                                        "Gd" : 4, "Ex" : 5},

                       "CentralAir":   {"None" : 0, "N" : 1, "Y" : 2},

                       "ExterCond":    {"None" : 0, "Po" : 1, "Fa" : 2, "TA": 3, 

                                        "Gd": 4, "Ex" : 5},

                       "ExterQual":    {"None" : 0, "Po" : 1, "Fa" : 2, "TA": 3, 

                                        "Gd": 4, "Ex" : 5},

                       "Fence":        {"None" : 0, "MnWw" : 1, "GdWo" : 2, "MnPrv": 3, 

                                        "GdPrv" : 4},

                       "FireplaceQu":  {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, 

                                        "Gd" : 4, "Ex" : 5},

                       "Functional":   {"None" : 0, "Sal" : 1, "Sev" : 2, "Maj2" : 3, 

                                        "Maj1" : 4, "Mod": 5, "Min2" : 6, "Min1" : 7, 

                                        "Typ" : 8},

                       "GarageCond":   {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, 

                                        "Gd" : 4, "Ex" : 5},

                       "GarageQual":   {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, 

                                        "Gd" : 4, "Ex" : 5},

                       "GarageFinish": {"None" : 0, "Unf" : 1, "RFn" : 2, "Fin" : 3},

                       "HeatingQC":    {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, 

                                        "Gd" : 4, "Ex" : 5},

                       "KitchenQual":  {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, 

                                        "Gd" : 4, "Ex" : 5},

                       "LandContour":  {"None" : 0, "Low" : 1, "HLS" : 2, "Bnk" : 3, 

                                        "Lvl" : 4},

                       "LandSlope":    {"None" : 0, "Sev" : 1, "Mod" : 2, "Gtl" : 3},

                       "LotShape":     {"None" : 0, "IR3" : 1, "IR2" : 2, "IR1" : 3, 

                                        "Reg" : 4},

                       "PavedDrive":   {"None" : 0, "N" : 0, "P" : 1, "Y" : 2},

                       "PoolQC":       {"None" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, 

                                        "Ex" : 4},

                       "Street":       {"None" : 0, "Grvl" : 1, "Pave" : 2},

                       "Utilities":    {"None" : 0, "ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, 

                                        "AllPub" : 4}}

                     )



feat.BsmtCond = feat.BsmtCond.astype(int)
# only one hot encode «true» categoricals... 

# ... rather than ordinals, where order matters and we already label encoded in the previous cells



def onehot_encode(data):

    df_numeric = data.select_dtypes(exclude=['object'])

    df_obj = data.select_dtypes(include=['object']).copy()



    cols = []

    for c in df_obj:

        dummies = pd.get_dummies(df_obj[c])

        dummies.columns = [c + "_" + str(x) for x in dummies.columns]

        cols.append(dummies)

    df_obj = pd.concat(cols, axis=1)



    data = pd.concat([df_numeric, df_obj], axis=1)

    data.reset_index(inplace=True, drop=True)

    return data



feat = onehot_encode(feat)
# map months to seasons: 0 == winter, 1 == spring etc.

seasons = {12 : 0, 1 : 0, 2 : 0, 

           3 : 1, 4 : 1, 5 : 1,

           6 : 2, 7 : 2, 8 : 2, 

           9 : 3, 10 : 3, 11 : 3}



feat["SeasonSold"]     = feat["MoSold"].map(seasons)

feat["YrActualAge"]    = feat["YrSold"] - feat["YearBuilt"]



feat['TotalSF1']       = feat['TotalBsmtSF'] + feat['1stFlrSF'] + feat['2ndFlrSF']

feat['TotalSF2']       = feat['BsmtFinSF1'] + feat['BsmtFinSF2'] + feat['1stFlrSF'] + feat['2ndFlrSF']

feat["AllSF"]          = feat["GrLivArea"] + feat["TotalBsmtSF"]

feat["AllFlrsSF"]      = feat["1stFlrSF"] + feat["2ndFlrSF"]

feat["AllPorchSF"]     = feat["OpenPorchSF"] + feat["EnclosedPorch"] + feat["3SsnPorch"] + feat["ScreenPorch"]



feat['TotalBath']      = 2 * (feat['FullBath'] + (0.5 * feat['HalfBath']) + feat['BsmtFullBath'] + (0.5 * feat['BsmtHalfBath']))



feat["TotalBath"]      = feat["TotalBath"].astype(int)

feat['TotalPorch']     = feat['OpenPorchSF'] + feat['3SsnPorch'] + feat['EnclosedPorch'] + feat['ScreenPorch'] + feat['WoodDeckSF']

feat["OverallScore"]   = feat["OverallQual"] * feat["OverallCond"]

feat["GarageScore"]    = feat["GarageQual"] * feat["GarageCond"]

feat["ExterScore"]     = feat["ExterQual"] * feat["ExterCond"]

feat["KitchenScore"]   = feat["KitchenAbvGr"] * feat["KitchenQual"]

feat["FireplaceScore"] = feat["Fireplaces"] * feat["FireplaceQu"]

feat["GarageScore"]    = feat["GarageArea"] * feat["GarageQual"]

feat["PoolScore"]      = feat["PoolArea"] * feat["PoolQC"]



feat['hasPool']        = feat['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

feat['has2ndFloor']    = feat['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

feat['hasGarage']      = feat['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

feat['hasBsmt']        = feat['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

feat['hasFireplace']   = feat['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
# create new ordinal features by binning continuous features

# log transform values before binning taking into account skewed distributions



cut_cols = ["LotArea", "YearBuilt", "YearRemodAdd", "MasVnrArea", 'BsmtFinSF1',

            'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',

            'LowQualFinSF', 'GrLivArea', "GarageYrBlt", 'GarageArea', 'WoodDeckSF', 'OpenPorchSF']

frames = []

for cut_col in cut_cols:

    tmp = pd.DataFrame(pd.cut(np.log1p(feat[cut_col]), bins=10, labels=np.arange(0,10)))

    tmp.columns = [cut_col + "_binned"]

    frames.append(tmp)

    

binned = pd.concat(frames, axis=1).astype(int)

feat = pd.concat([feat, binned], axis=1)
dtrain = feat[feat.SalePrice.notnull()].copy()

dtest  = feat[feat.SalePrice.isnull()].copy()

dtest  = dtest.drop("SalePrice", axis=1).reset_index(drop=True)

print(f"Raw data shape   : {df.shape}  {df_test.shape}")

print(f"Clean data shape : {dtrain.shape} {dtest.shape}")
# X = dtrain[dtrain.GrLivArea < 4000].drop(["SalePrice"], axis=1)

# y = np.log1p(dtrain[dtrain.GrLivArea < 4000].SalePrice)

# metric = 'neg_mean_squared_error'



# sk = pd.DataFrame(X.iloc[:, :60].skew(), columns=["skewness"])

# sk = sk[sk.skewness > .75]

# for feature_ in sk.index:

#     X[feature_] = boxcox1p(X[feature_], 0.15)
# # GridSearchCV Ridge

# ridge = make_pipeline(RobustScaler(), Ridge(alpha=15, random_state=1))

# param_grid = {

#     'ridge__alpha' : np.linspace(12, 18, 10),

#     'ridge__max_iter' : np.linspace(10, 200, 5),

# }

# search = GridSearchCV(ridge, param_grid, cv=5, scoring=metric, n_jobs=-1)

# search.fit(X, y)

# print(f"{search.best_params_}")

# print(f"{np.sqrt(-search.best_score_):.4}")
# # GridSearchCV Lasso

# lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.00044, random_state=1))

# param_grid = {'lasso__alpha' : np.linspace(0.00005, 0.001, 30)}

# search = GridSearchCV(lasso, param_grid, cv=5, scoring=metric, n_jobs=-1)

# search.fit(X, y)

# print(f"{search.best_params_}")

# print(f"{np.sqrt(-search.best_score_):.4}")
# # GridSearchCV ElasticNet

# elastic = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=1, random_state=1))

# param_grid = {

#     'elasticnet__alpha' : np.linspace(0.0001, 0.001, 10),

#     'elasticnet__l1_ratio' : np.linspace(0.5, 1, 10),

# }

# search = GridSearchCV(elastic, param_grid, cv=5, scoring=metric, n_jobs=-1)

# search.fit(X, y)

# print(f"{search.best_params_}")

# print(f"{np.sqrt(-search.best_score_):.4}")
# # GridSearchCV KernelRidge

# kernel = KernelRidge(alpha=1)

# param_grid = {'alpha' : np.linspace(0.001, 1, 30)}

# search = GridSearchCV(kernel, param_grid, cv=5, scoring=metric, n_jobs=-1)

# search.fit(X, y)

# print(f"{search.best_params_}")

# print(f"{np.sqrt(-search.best_score_):.4}")
# # GridSearchCV GBM

# # huber loss is considered less sensitive to outliers

# gbm = GradientBoostingRegressor(n_estimators=2500, learning_rate=0.04,

#                                    max_depth=2, max_features='sqrt',

#                                    min_samples_leaf=15, min_samples_split=10, 

#                                    loss='huber', random_state=1)

# param_grid = {

#     'n_estimators' : [2500],

#     'learning_rate' : [0.03, 0.04, 0.05],

#     'max_depth' : [2],

# }

# search = GridSearchCV(gbm, param_grid, cv=5, scoring=metric, n_jobs=-1)

# search.fit(X, y)

# print(f"{search.best_params_}")

# print(f"{np.sqrt(-search.best_score_):.4}")
# # GridSearchCV LightGBM

# lgbm = lgb.LGBMRegressor(objective='regression', num_leaves=5,

#                         learning_rate=0.03, n_estimators=8000,

#                         max_bin=55, bagging_fraction=0.8,

#                         bagging_freq=5, feature_fraction=0.23,

#                         feature_fraction_seed=9, bagging_seed=9,

#                         min_data_in_leaf=6, min_sum_hessian_in_leaf=11)

# param_grid = {

#     'n_estimators' : [8000],

#     'learning_rate' : [0.03],

# }

# search = GridSearchCV(clf, param_grid, cv=5, scoring=metric, n_jobs=-1)

# search.fit(X, y)

# print(f"{search.best_params_}")

# print(f"{np.sqrt(-search.best_score_):.4}")
# # GridSearchCV XGBoost

# xgbreg = xgb.XGBRegressor(objective="reg:squarederror",

#                              colsample_bytree=0.46, gamma=0.047, 

#                              learning_rate=0.04, max_depth=2, 

#                              min_child_weight=0.5, n_estimators=2000,

#                              reg_alpha=0.46, reg_lambda=0.86,

#                              subsample=0.52, random_state=1, n_jobs=-1)



# param_grid = {

#     'xgbregressor__max_depth' : [2],

#     'xgbregressor__estimators' : [1600, 1800, 2000],

#     "xgbregressor__learning_rate" : [0.02, 0.03, 0.04],

#     "xgbregressor__min_child_weight" : [0.2, 0.3, 0.4],

#     }

# search = GridSearchCV(clf, param_grid, cv=3, scoring=metric, n_jobs=-1)

# search.fit(X, y)

# print(f"{search.best_params_}")

# print(f"{np.sqrt(-search.best_score_):.4}")
# # try a stacked regressor on top of the seven tuned classifiers 

# # leaving out xgboost in the stack for now since it seems to crash the stacked regressor

# clf_to_stack = [lasso, ridge, elastic, kernel, gbm, lgbm]



# stack = StackingCVRegressor(regressors=(clf_to_stack),

#                             meta_regressor=xgb.XGBRegressor(objective="reg:squarederror", n_jobs=-1), 

#                             use_features_in_secondary=True)



# print(f"{np.sqrt(-cross_val_score(stack, X, y, scoring=metric)).mean():.4f} Log Error")
X = dtrain[dtrain.GrLivArea < 4000].drop(["SalePrice"], axis=1)

y = np.log1p(dtrain[dtrain.GrLivArea < 4000].SalePrice)

X_test = dtest

metric = 'neg_mean_squared_error'
# apply box cox transformation on numerical features

# skipping the one hot encoded features as well as engineered ones

sk = pd.DataFrame(X.iloc[:, :60].skew(), columns=["skewness"])

sk = sk[sk.skewness > .75]

for feature_ in sk.index:

    X[feature_] = boxcox1p(X[feature_], 0.15)

    X_test[feature_] = boxcox1p(X_test[feature_], 0.15)
ridge   = make_pipeline(RobustScaler(), Ridge(alpha=15, random_state=1))

lasso   = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))

elastic = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, 

                                                   l1_ratio=1, random_state=1))

kernel  = KernelRidge(alpha=1.0)



gbm = GradientBoostingRegressor(n_estimators=2500, learning_rate=0.04,

                                   max_depth=2, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state=1)



lgbm = lgb.LGBMRegressor(objective='regression', num_leaves=5,

                        learning_rate=0.03, n_estimators=8000,

                        max_bin=55, bagging_fraction=0.8,

                        bagging_freq=5, feature_fraction=0.23,

                        feature_fraction_seed=9, bagging_seed=9,

                        min_data_in_leaf=6, min_sum_hessian_in_leaf=11)



xgbreg = xgb.XGBRegressor(objective="reg:squarederror",

                             colsample_bytree=0.46, gamma=0.047, 

                             learning_rate=0.04, max_depth=2, 

                             min_child_weight=0.5, n_estimators=2000,

                             reg_alpha=0.46, reg_lambda=0.86,

                             subsample=0.52, random_state=1, n_jobs=-1)
classifiers = [ridge, lasso, elastic, kernel, gbm, lgbm, xgbreg]

clf_names   = ["ridge  ", "lasso  ", "elastic", "kernel ", "gbm    ", "lgbm   ", "xgbreg "]



predictions_exp = []



for clf_name, clf in zip(clf_names, classifiers):

    print(f"{clf_name} {np.sqrt(-cross_val_score(clf, X, y, scoring=metric).mean()):.5f}")

    clf.fit(X, y)

    preds = clf.predict(X_test)

    predictions_exp.append(np.expm1(preds))
# option to set differents weights to models



weighted = (1 *   predictions_exp[0] + \

            1 *   predictions_exp[1] + \

            1 *   predictions_exp[2] + \

            1 *   predictions_exp[3] + \

            1 *   predictions_exp[4] + \

            1 *   predictions_exp[5] + \

            1 *   predictions_exp[6]) / 7
prediction_final = pd.DataFrame(weighted, columns=["SalePrice"])

submission = pd.DataFrame({'Id': df_test.Id.values, 'SalePrice': prediction_final.SalePrice.values})

submission.to_csv(f"submission.csv", index=False)