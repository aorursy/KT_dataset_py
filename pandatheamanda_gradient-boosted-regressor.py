import numpy as np

import pandas as pd

import matplotlib.pylab as plt

import sklearn

from sklearn.impute import SimpleImputer

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from mlxtend.regressor import StackingRegressor

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.svm import SVR

from sklearn.linear_model import Lasso

from sklearn.pipeline import make_pipeline

from sklearn.pipeline import Pipeline

from mlxtend.regressor import StackingCVRegressor

train_set = pd.read_csv("../input/train.csv")

test_set = pd.read_csv("../input/test.csv")

train_set.drop(train_set[(train_set['GrLivArea']>4000) & (train_set['SalePrice']<300000)].index, inplace=True)

train_set.drop(train_set[(train_set['GarageArea']>1200) & (train_set['SalePrice']<100000)].index, inplace=True)

train_set.drop(train_set[(train_set['TotalBsmtSF']>6000) & (train_set['SalePrice']<200000)].index, inplace=True)

y_train = train_set.SalePrice



train_set = train_set.drop(["SalePrice"], axis=1)

train_set = train_set.set_index('Id')

test_set = test_set.set_index('Id')

weak=['BedroomAbvGr', 'ScreenPorch', 'PoolArea', 'MoSold', '3SsnPorch',

       'BsmtFinSF2', 'BsmtHalfBath', 'MiscVal', 'LowQualFinSF', 'YrSold',

       'OverallCond', 'MSSubClass', 'EnclosedPorch', 'KitchenAbvGr']

train_set.drop(weak, axis=1, inplace=True)

test_set.drop(weak, axis=1, inplace=True)
firstmerge = pd.concat([train_set, test_set], axis=0)

data_dummies= pd.get_dummies(firstmerge)

data_dummies = data_dummies.reset_index(drop=True)

train_set = data_dummies.loc[0:1456]

test_set = data_dummies.loc[1457:]

my_imputer = SimpleImputer()

train_set = my_imputer.fit_transform(train_set)

test_set = my_imputer.transform(test_set)

scaler = MinMaxScaler()

scaler.fit(train_set)

train_scaled = scaler.transform(train_set)

test_scaled = scaler.transform(test_set)



y_train = np.log(y_train)


# param_grid = {'learning_rate':[0.05, 0.02, 0.01],

# 'max_depth': [3],

# 'min_samples_leaf': [3],



# }

# est = GradientBoostingRegressor(n_estimators=5000, random_state=33, )

# gs_cv = GridSearchCV(est, param_grid, n_jobs=4, cv=5).fit(train_scaled, y_train)

# gs_cv.best_score_

# gbr = GradientBoostingRegressor(n_estimators=5000,

#                                    max_depth=3, max_features='sqrt',

#                                    min_samples_leaf=5, min_samples_split=10, 

#                                    loss='huber', random_state =5)

#GBoost.fit(train_scaled, y_train)
ridge = Ridge()

lasso = Lasso()

svr_rbf = SVR(kernel='rbf')

rf = RandomForestRegressor()

gb = GradientBoostingRegressor(max_depth=3, max_features='sqrt', loss='huber', 

                               min_samples_leaf=10, min_samples_split=10,

                              n_estimators=10000, learning_rate=.01)



stack = StackingCVRegressor(regressors=(lasso, ridge, svr_rbf, rf),

                            meta_regressor=gb, 

                            use_features_in_secondary=True)



pipeline = make_pipeline(stack)



params = {'stackingcvregressor__lasso__alpha': [0.1, 1.0],

          'stackingcvregressor__ridge__alpha': [0.1, 1.0],

         'stackingcvregressor__svr__C': [0.1, 1.0]}



grid = GridSearchCV(

    verbose=1,

    estimator=pipeline, 

    param_grid=params, 

    cv=5,

    refit=True

)



grid.fit(train_scaled, y_train)
# lr = LinearRegression()

# ridge = Ridge(random_state=1)

# svr_rbf = SVR(kernel='rbf')

# lasso = Lasso(random_state=1)



# stack = StackingCVRegressor(regressors=(lr, ridge, svr_rbf, lasso),

#                             meta_regressor=gbr, 

#                             use_features_in_secondary=True)

# pipeline = make_pipeline(stack)

# params = {'stackingcvregressor_lasso__alpha': [0.1, 1.0],

#           'stackingcvregressor_ridge__alpha': [0.1, 1.0],

#           'stackingcvregressor_svr__C': [0.1, 1.0],

#           'stackingcvregressor_svr__C': [0.1, 1.0],

#           'stackingcvregressor_svr__gamma': [0.1, 1.0]

#          }





# # stregr = StackingRegressor(regressors=[svr_lin, lr, ridge, lasso, svr_rbf], 

# #                            meta_regressor=gbr)

# grid = GridSearchCV(estimator=pipeline, 

#                     param_grid=params, 

#                     cv=5,

#                     refit=True)

# grid.fit(train_scaled, y_train)
test_result = grid.predict(test_scaled)

test_result = np.exp(test_result)-1 

df= pd.DataFrame({'SalePrice': test_result}) 

df.index.name='Id' 

df.index +=1461 

df.to_csv('gridsearch.csv')