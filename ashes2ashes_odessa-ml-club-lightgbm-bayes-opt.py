# Code you have previously used to load data

import matplotlib.pyplot as plt

import mlxtend

import numpy as np

import pandas as pd



from bayes_opt import BayesianOptimization

from datetime import datetime as dt

from lightgbm import LGBMRegressor

from sklearn.base import clone

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import ElasticNet, ElasticNetCV

from sklearn.metrics import mean_squared_error, mean_squared_log_error

from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, train_test_split

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import Imputer, RobustScaler, StandardScaler

from sklearn.tree import DecisionTreeRegressor

from warnings import filterwarnings

from xgboost import XGBRegressor



filterwarnings('ignore')
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

print(df_train.shape)

print(df_test.shape)

n_train = df_train.shape[0]



target = df_train['SalePrice']

df_train.drop(columns=['SalePrice', 'Id'], inplace=True)

test_id = df_test['Id']

df_test.drop(columns=['Id'], inplace=True)

assert((df_train.columns == df_test.columns).all())



df_total = pd.concat([df_train, df_test])

df_numeric = df_total.select_dtypes(include='number')

df_onehot = pd.get_dummies(df_total)

print(df_numeric.shape[1], 'vs', df_onehot.shape[1])
plt.title('Target')

plt.hist(target)

plt.hist(target, 100);
target_log = np.log(target)

X_train, X_test, y_train, y_test = train_test_split(df_onehot[:n_train], target_log, test_size=0.1, random_state=42)

print(np.mean(y_train), np.median(y_train))
def root_mean_squared_error(y_true, y_pred):

    return np.sqrt(mean_squared_error(y_true, y_pred))





def root_mean_squared_log_error(y_true, y_pred):

     return np.sqrt(mean_squared_log_error(y_true, y_pred))





def RMSLE(estimator, X, y):

    return root_mean_squared_error(np.log(y + 1), np.log(estimator.predict(X) + 1))





def negRMSLE(estimator, X, y):

    return -RMSLE(estimator, X, y)





def RMSE(estimator, X, y):

    return root_mean_squared_error(y, estimator.predict(X))





def negRMSE(estimator, X, y):

    return -RMSE(estimator, X, y)





def elastic_baseline():

    pipe = Pipeline([

        ('imputer', Imputer()),

        ('scaler', RobustScaler()),

        ('regressor', ElasticNet(random_state=42))

    ])

    params = {

        'imputer__strategy': ['mean', 'median'],

        'regressor__alpha': [10**i for i in range(-3, 4)],

        'regressor__l1_ratio': [0.0, 0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

    }

    grid = GridSearchCV(pipe, params, scoring='neg_mean_squared_error', cv=10, n_jobs=-1, return_train_score=True, verbose=2)

    grid.fit(X_train, y_train)

    

#     df_scores = pd.DataFrame(grid.cv_results_)

    print('Best CV-score: {:.5f}'.format(grid.best_score_))

    print('Best params:', grid.best_params_)

    print('Time to refit on whole X_train: {:.2f}s'.format(grid.refit_time_))



    model_baseline = pipe

    model_baseline.fit(X_train, y_train)

    print('Baseline train score: {:.5f}'.format(RMSE(model_baseline, X_train, y_train)))

    print('Baseline test score: {:.5f}'.format(RMSE(model_baseline, X_test, y_test)))



    model = grid.best_estimator_

    model.fit(X_train, y_train)

    print('Optimized train score: {:.5f}'.format(RMSE(model, X_train, y_train)))

    print('Optimized test score: {:.5f}'.format(RMSE(model, X_test, y_test)))
# elastic_baseline()
def make_model(method, **kwargs):

    if method == 'xgb':

        model_baseline = XGBRegressor(random_state=42, n_jobs=-1)

    elif method == 'lgb':

        model_baseline = LGBMRegressor(random_state=42, n_jobs=-1)

    elif method == 'elnet':

        model_baseline = Pipeline([

            ('imputer', Imputer()),

            ('scaler', StandardScaler()),

            ('regressor', ElasticNet(random_state=42))

        ])

    model = clone(model_baseline).set_params(**kwargs)

    return model, model_baseline





def process_params(method, **kwargs):

    kw = dict()

    if method == 'xgb' or method == 'lgb':

        kw['n_estimators'] = int(kwargs['n_estimators'] + 0.5)

        kw['max_depth'] = int(kwargs['max_depth'] + 0.5)

        kw['num_leaves'] = int(kwargs['num_leaves'] + 0.5)

        kw['min_child_samples'] = int(kwargs['min_child_samples'] + 0.5)

        kw['learning_rate'] = 10**kwargs['learning_rate']

        kw['reg_alpha'] = 10**kwargs['reg_alpha']

        kw['reg_lambda'] = 10**kwargs['reg_lambda']

        kw['subsample'] = kwargs['subsample']

        kw['colsample_bytree'] = kwargs['colsample_bytree']

    elif method == 'elnet':

        kw['imputer__strategy'] = 'mean' if kwargs['imputer__strategy'] <= 0.5 else 'median'

        kw['regressor__alpha'] = 10**kwargs['regressor__alpha']

        kw['regressor__l1_ratio'] = kwargs['regressor__l1_ratio']

    return kw





def get_cv_score(X, y, method, cv=10, **kwargs):

    kwargs = process_params(method, **kwargs)

    model, _ = make_model(method, **kwargs)

    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)

    return -np.sqrt(-scores.mean())





def optimize(X, y, method, n_iter=15, init_points=5, kappa=10):

    def fun(**kwargs): 

        return get_cv_score(

            X,

            y,

            method,

            **kwargs

        )





    if method == 'xgb' or method == 'lgb':

        pbounds={

            'n_estimators': (100, 1000),

            'max_depth': (3, 30),

            'num_leaves': (8, 64),

            'min_child_samples': (8, 64),

            'learning_rate': (-2, -1),

            'reg_alpha': (-3, 3),

            'reg_lambda': (-3, 3),

            'subsample': (0.1, 1),

            'colsample_bytree': (0.1, 1)

        }

    elif method == 'elnet':

        pbounds={

            'imputer__strategy': (0, 1),

            'regressor__alpha': (-3, 3),

            'regressor__l1_ratio': (0, 1)

        }

    

    

    optimizer = BayesianOptimization(fun, pbounds, random_state=42)

    optimizer.maximize(n_iter=n_iter, init_points=init_points, kappa=kappa)

    print('\tFinal result:\n', optimizer.max)

    return optimizer





def test_model(method, thresh=0, show_cols=False, **best_params):

    def print_cols(cols):

        print('[', end='')

        for col in cols:

            print("'{}'".format(col), end=', ')

        print(']')

    

    model, model_baseline = make_model(method, **best_params)

    

    model_baseline.fit(X_train, y_train)

    print('Baseline train score: {:.5f}'.format(RMSE(model_baseline, X_train, y_train)))

    print('Baseline test score: {:.5f}'.format(RMSE(model_baseline, X_test, y_test)))

          

    model.fit(X_train, y_train)

    print('Optimized train score: {:.5f}'.format(RMSE(model, X_train, y_train)))

    print('Optimized test score: {:.5f}'.format(RMSE(model, X_test, y_test)))

    

    if model == 'elnet':

        cols = X_train.columns

    else:

        plt.hist(np.array(model.feature_importances_))

        plt.hist(np.array(model.feature_importances_), bins=100)

        imp = pd.Series(index=X_train.columns, data=model.feature_importances_)

        imp = imp.sort_values(ascending=False)

        cols = list((imp[imp > thresh]).index)

        if show_cols:

            print_cols(cols)

    

    return cols
opt = optimize(X_train, y_train, 'lgb')

print('Best CV-score: {:.5f}'.format(opt.max['target']))

best_params = process_params('lgb', **opt.max['params'])

print('Best params:', best_params)

# best_params = params = {'n_estimators': 635, 'max_depth': 4, 'num_leaves': 61, 'min_child_samples': 8, 'subsample': 0.6181601499844832,

#           'colsample_bytree': 0.17834063704632525, 'learning_rate': 0.04238876990190508, 'reg_alpha': 0.015776461541004157,

#           'reg_lambda': 0.00102918700714727}

cols = test_model('lgb', 0, **best_params)
def make_submission(model, df, target, use_exp=False, cols=None):

    if cols is None:

        cols = df.columns

    model.fit(df[:n_train][cols], target)

    test_pred = model.predict(df[n_train:][cols])

    if use_exp:

        test_pred = np.exp(test_pred)

    output = pd.DataFrame({'Id': test_id, 'SalePrice': test_pred})

    now = dt.now()

    fname = 'subm_{}_{}__{}_{}.csv'.format(now.day, now.month, now.hour, now.minute)

    output.to_csv(fname, index=False)
# model = Pipeline([

#         ('imputer', Imputer(strategy='median')),

#         ('scaler', StandardScaler()),

#         ('regressor', ElasticNet(alpha=1.0, l1_ratio=0.3, random_state=42))

#     ])

# make_submission(model, df_onehot)



model = Pipeline([

    ('imputer', Imputer()),

    ('scaler', StandardScaler()),

    ('regressor', ElasticNet(random_state=42))

])

params = {'imputer__strategy': 'mean', 'regressor__alpha': 0.1, 'regressor__l1_ratio': 0.05}

model.set_params(**params)

make_submission(model, df_onehot, target_log, use_exp=True)



# params = {'n_estimators': 975, 'max_depth': 22, 'num_leaves': 62, 'min_child_samples': 28, 'subsample': 0.37079047883509275,

#           'colsample_bytree': 0.3908826388186797, 'learning_rate': 0.010903884523201108, 'reg_alpha': 0.03241110030999171,

#           'reg_lambda': 0.9627001408471512}

# cols1 = ['GrLivArea', 'TotalBsmtSF', 'LotArea', '1stFlrSF', 'GarageArea', 'BsmtFinSF1', 'LotFrontage', 'GarageYrBlt', 'BsmtUnfSF', 'YearBuilt', 'YearRemodAdd', 'OpenPorchSF', '2ndFlrSF', 'MasVnrArea', 'OverallQual', 'WoodDeckSF', 'OverallCond', 'MoSold', 'TotRmsAbvGrd', 'YrSold', 'MSSubClass', 'Fireplaces', 'GarageCars', 'BedroomAbvGr', 'BsmtFullBath', 'FullBath', 'BsmtExposure_No', 'HalfBath', 'SaleCondition_Normal', 'LandContour_Bnk', 'LotShape_Reg', 'BsmtQual_Gd', 'FireplaceQu_Gd', 'Functional_Typ', 'GarageType_Attchd', 'HouseStyle_1Story', 'BsmtFinType1_GLQ', 'Neighborhood_Crawfor', 'KitchenQual_Gd', 'KitchenQual_TA', 'BsmtExposure_Gd', 'SaleType_New', 'ScreenPorch', 'HeatingQC_Ex', 'LotShape_IR1', 'HouseStyle_2Story', 'EnclosedPorch', 'Neighborhood_NoRidge', 'GarageType_Detchd', 'BsmtFinType1_ALQ', 'Exterior1st_BrkFace', 'HeatingQC_TA', 'ExterQual_TA', 'MasVnrType_BrkFace', 'Condition1_Norm', 'MSZoning_RL', 'SaleCondition_Abnorml', 'BsmtFinSF2', 'LandContour_Lvl', 'CentralAir_N', 'GarageFinish_Unf', 'MSZoning_RM', 'BsmtQual_Ex', 'GarageFinish_RFn', 'ExterQual_Gd', 'KitchenQual_Ex', 'Neighborhood_Somerst', 'LotConfig_CulDSac', 'RoofStyle_Gable', 'Neighborhood_Edwards', 'BsmtQual_TA', 'BsmtFinType1_Unf', 'Neighborhood_NAmes', 'Exterior1st_VinylSd', 'Condition1_Artery', 'ExterCond_TA', 'SaleCondition_Partial', 'Foundation_PConc', 'Neighborhood_BrkSide', 'Neighborhood_OldTown', 'MasVnrType_None', 'KitchenAbvGr', 'BsmtCond_Fa', 'BsmtCond_TA', 'GarageFinish_Fin', 'CentralAir_Y', 'LandSlope_Gtl', 'GarageCond_TA', 'LotConfig_Inside', 'SaleType_WD', 'Foundation_CBlock', 'Exterior1st_MetalSd', 'FireplaceQu_TA', 'RoofStyle_Hip', 'Exterior2nd_VinylSd', 'PavedDrive_Y', 'Exterior1st_HdBoard', 'GarageQual_TA', 'ExterQual_Ex', 'Exterior1st_Plywood', 'Fence_GdWo', 'Foundation_BrkTil', 'BsmtExposure_Av', 'Exterior1st_Wd Sdng', 'MSZoning_FV', 'BldgType_1Fam', 'Exterior2nd_Wd Sdng', 'Exterior2nd_MetalSd', ]



# params = {'n_estimators': 763, 'max_depth': 4, 'num_leaves': 62, 'min_child_samples': 8, 'subsample': 0.2936474414836979,

#           'colsample_bytree': 0.13176165663335335, 'learning_rate': 0.03292552347864284, 'reg_alpha': 0.02023613728543995,

#           'reg_lambda': 0.008391229998184617}

# params = {'n_estimators': 635, 'max_depth': 4, 'num_leaves': 61, 'min_child_samples': 8, 'subsample': 0.6181601499844832,

#           'colsample_bytree': 0.17834063704632525, 'learning_rate': 0.04238876990190508, 'reg_alpha': 0.015776461541004157,

#           'reg_lambda': 0.00102918700714727}

# model = LGBMRegressor(random_state=42, n_jobs=-1)



# model.set_params(**params)

# make_submission(model, df_onehot, cols)