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
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

from scipy import stats

from sklearn.metrics import accuracy_score
train = pd.read_csv("../input/train.csv")

print("The shape of train data : " + str(train.shape))
test = pd.read_csv("../input/test.csv")

print("The shape of test data : " + str(test.shape))
train.head()
train.info()
test.info()
plt.subplots(figsize = (12,10))

sns.distplot(train['SalePrice'], fit = stats.norm)



fig = plt.figure()

stats.probplot(train['SalePrice'], plot = plt)

plt.show()
train['SalePrice'] = np.log1p(train['SalePrice'])
plt.subplots(figsize = (12,10))

sns.distplot(train['SalePrice'], fit = stats.norm)



fig = plt.figure()

stats.probplot(train['SalePrice'], plot = plt)

plt.show()
plt.scatter(train.GrLivArea, train.SalePrice, c = "blue", marker = "o")

plt.title("Looking for outliers")

plt.xlabel("GrLivArea")

plt.ylabel("SalePrice")

plt.show()
train.columns[train.isnull().any()]
plt.subplots(figsize = (12,10))

sns.heatmap(train.isnull())

plt.show()
train_id = train['Id']

train.drop(['Id'], axis = 1, inplace = True)

y_train = train['SalePrice']
test_id = test['Id']

test.drop(['Id'], axis = 1, inplace = True)
train_corr = train.corr()

plt.subplots(figsize = (12,10))

sns.heatmap(train_corr)

plt.show()
new_train = train.shape[0]

new_test = test.shape[0]

y_train = train.SalePrice.values

all_data = pd.concat((train, test)).reset_index(drop=True)

all_data.drop(['SalePrice'], axis=1, inplace=True)

print("The shape of all_data is : {}".format(all_data.shape))
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Percentage' :all_data_na})

missing_data.head(30)
all_data['PoolQC'] = all_data['PoolQC'].fillna('None')

all_data['MiscFeature'] = all_data['MiscFeature'].fillna('None')

all_data['Alley'] = all_data['Alley'].fillna('None')

all_data['Fence'] = all_data['Fence'].fillna('None')

all_data['FireplaceQu'] = all_data['FireplaceQu'].fillna('None')

all_data['LotFrontage'] = all_data['LotFrontage'].fillna('None')

all_data['Neighborhood'] = all_data['Neighborhood'].fillna('None')



for col in ['GarageCond', 'GarageType','GarageQual','GarageYrBlt','GarageFinish']:

    all_data[col] = all_data[col].fillna('None')



for col in ['BsmtExposure','BsmtFinType2','BsmtFinType1','BsmtCond','BsmtQual','MasVnrType']:

    all_data[col] = all_data[col].fillna('None')



all_data['MasVnrArea'] = all_data['MasVnrArea'].fillna(int(0))

all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['MSZoning'] = all_data['MSZoning'].fillna('None')

all_data['Functional'] = all_data['Functional'].fillna('None')

all_data['BsmtFullBath'] = all_data['BsmtFullBath'].fillna('None')

all_data['BsmtHalfBath'] = all_data['BsmtHalfBath'].fillna('None')

all_data['TotalBsmtSF'] = all_data['TotalBsmtSF'].fillna('None')

all_data['SaleType'] =all_data['SaleType'].fillna('None')

all_data['GarageArea'] = all_data['GarageArea'].fillna('None')

all_data['Exterior1st'] = all_data['Exterior1st'].fillna('None')

all_data['BsmtUnfSF'] = all_data['BsmtUnfSF'].fillna('None')

all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna('None')

all_data['KitchenQual'] = all_data['KitchenQual'].fillna('None')

all_data['GarageCars'] = all_data['GarageCars'].fillna('None')

all_data['BsmtFinSF1'] =all_data['BsmtFinSF1'].fillna('None')

all_data['BsmtFinSF2'] = all_data['BsmtFinSF2'].fillna('None')

                                                       

all_data['Utilities'] = all_data.drop(['Utilities'], axis = 1)
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Percentage' :all_data_na})

missing_data.head(30)
from sklearn.preprocessing import LabelEncoder

cols = ('1stFlrSF', '2ndFlrSF', '3SsnPorch', 'Alley', 'BedroomAbvGr',

       'BldgType', 'BsmtCond', 'BsmtExposure', 'BsmtFinSF1', 'BsmtFinSF2',

       'BsmtFinType1', 'BsmtFinType2', 'BsmtFullBath', 'BsmtHalfBath',

       'BsmtQual', 'BsmtUnfSF', 'CentralAir', 'Condition1', 'Condition2',

       'Electrical', 'EnclosedPorch', 'ExterCond', 'ExterQual', 'Exterior1st',

       'Exterior2nd', 'Fence', 'FireplaceQu', 'Fireplaces', 'Foundation',

       'FullBath', 'Functional', 'GarageArea', 'GarageCars', 'GarageCond',

       'GarageFinish', 'GarageQual', 'GarageType', 'GarageYrBlt', 'GrLivArea',

       'HalfBath', 'Heating', 'HeatingQC', 'HouseStyle', 'KitchenAbvGr',

       'KitchenQual', 'LandContour', 'LandSlope', 'LotArea', 'LotConfig',

       'LotFrontage', 'LotShape', 'LowQualFinSF', 'MSSubClass', 'MSZoning',

       'MasVnrArea', 'MasVnrType', 'MiscFeature', 'MiscVal', 'MoSold',

       'Neighborhood', 'OpenPorchSF', 'OverallCond', 'OverallQual',

       'PavedDrive', 'PoolArea', 'PoolQC', 'RoofMatl', 'RoofStyle',

       'SaleCondition', 'SaleType', 'ScreenPorch', 'Street', 'TotRmsAbvGrd',

       'TotalBsmtSF', 'Utilities', 'WoodDeckSF', 'YearBuilt', 'YearRemodAdd',

       'YrSold')



for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(all_data[c].values)) 

    all_data[c] = lbl.transform(list(all_data[c].values))



     

print('Shape all_data: {}'.format(all_data.shape))
x_train = all_data[:new_train]

x_test = all_data[new_train:]
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x_train,y_train)

lr_pred = lr.predict(x_test)



from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = lr, X = x_train, y = y_train, cv = 10)

accuracies.mean()
from sklearn.linear_model import LassoCV

lasso = LassoCV(alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 

                          0.3, 0.6, 1], 

                max_iter = 50000, cv = 10)

lasso.fit(x_train, y_train)

alpha = lasso.alpha_

print("Best alpha :", alpha)



print("Try again for more precision with alphas centered around " + str(alpha))

lasso = LassoCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, 

                          alpha * .85, alpha * .9, alpha * .95, alpha, alpha * 1.05, 

                          alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, alpha * 1.35, 

                          alpha * 1.4], 

                max_iter = 50000, cv = 10)

lasso.fit(x_train, y_train)

alpha = lasso.alpha_

print("Best alpha :", alpha)

lasso_pred = lasso.predict(x_test)
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = lasso, X = x_train, y = y_train, cv = 10)

accuracies.mean()
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators= 1000)

rfr.fit(x_train,y_train)

rfr_pred = rfr.predict(x_test)



from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = rfr, X = x_train, y = y_train, cv = 10)

accuracies.mean()
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor()

gbr.fit(x_train,y_train)

gbr_pred = gbr.predict(x_test)



from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = gbr, X = x_train, y = y_train, cv = 10)

accuracies.mean()
from sklearn.model_selection import GridSearchCV



param_grid={'n_estimators':[100], 'learning_rate': [0.1,0.05, 0.02, 0.01], 'max_depth':[6,4,6], 'min_samples_leaf':[3,5,9,17], 'max_features':[1.0,0.3,0.1] }

n_jobs=4 

grid = GridSearchCV(GradientBoostingRegressor(),param_grid)

grid.fit(x_train,y_train)
grid.best_params_
gbr = GradientBoostingRegressor(learning_rate= 0.1,

 max_depth= 6,

 max_features= 0.1,

 min_samples_leaf= 9,

 n_estimators= 100)

gbr.fit(x_train,y_train)

gbr_pred = gbr.predict(x_test)



from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = gbr, X = x_train, y = y_train, cv = 10)

accuracies.mean()
from xgboost import XGBRegressor 

xgbr = XGBRegressor(max_depth=4,learning_rate=0.1,n_estimators=1000)

xgbr.fit(x_train,y_train)

xgbr_pred = xgbr.predict(x_test)



from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = xgbr, X = x_train, y = y_train, cv = 10)

accuracies.mean()
param = {'nthread':[4], 

              'objective':['reg:linear'],

              'learning_rate': [.03, 0.05, .07],

              'max_depth': [5, 6, 7],

              'min_child_weight': [4],

              'silent': [1],

              'subsample': [0.7],

              'colsample_bytree': [0.7],

              'n_estimators': [500]}

grid = GridSearchCV(XGBRegressor(),param)

grid.fit(x_train,y_train)
grid.best_params_
xgbr = XGBRegressor(colsample_bytree= 0.7,

 learning_rate= 0.05,

 max_depth= 5,

 min_child_weight= 4,

 n_estimators=500,

 nthread=4,

 objective= 'reg:linear',

 silent= 1,

 subsample=0.7)

xgbr.fit(x_train,y_train)

xgbr_pred = xgbr.predict(x_test)
accuracies = cross_val_score(estimator = xgbr, X = x_train, y = y_train, cv = 10)

accuracies.mean()
from lightgbm import LGBMRegressor

lgbm = LGBMRegressor(objective='regression', 

                                       num_leaves=6,

                                       learning_rate=0.01, 

                                       n_estimators=6400,

                                       verbose=-1,

                                       bagging_fraction=0.80,

                                       bagging_freq=4, 

                                       bagging_seed=6,

                                       feature_fraction=0.2,

                                       feature_fraction_seed=7,

                                    )

lgbm.fit(x_train,y_train)

lgbm_pred = lgbm.predict(x_test)



accuracies = cross_val_score(estimator = lgbm, X = x_train, y = y_train, cv = 10)

accuracies.mean()
def rmse(y_train, y_pred):

    return np.sqrt(mean_squared_error(np.log(y_train), np.log(y_pred)))
from vecstack import stacking

from sklearn.metrics import r2_score, mean_squared_error

models = [lr,lasso,rfr,gbr,xgbr,lgbm]



S_train, S_test = stacking(models,

                           x_train, y_train, x_test,

                           regression=True,

                           mode='oof_pred_bag',

                           metric=rmse,

                           n_folds=5,

                           random_state=25,

                           verbose=2

                          )
xgbr_new =  XGBRegressor(colsample_bytree= 0.7,

 learning_rate= 0.05,

 max_depth= 5,

 min_child_weight= 4,

 n_estimators=500,

 nthread=4,

 objective= 'reg:linear',

 silent= 1,

 subsample=0.7)

xgbr_new.fit(S_train,y_train)

xgbr_new_pred = xgbr_new.predict(S_test)



accuracies = cross_val_score(estimator = xgbr_new, X = S_train, y = y_train, cv = 10)

accuracies.mean()
lr_pred1 = models[0].predict(x_test)

lasso_pred1 = models[1].predict(x_test)

rfr_pred1 = models[2].predict(x_test)

gbr_pred1 = models[3].predict(x_test)

xgbr_pred1 = models[4].predict(x_test)

lgbm_pred1 = models[5].predict(x_test)

S_test = np.c_[lr_pred1,lasso_pred1,rfr_pred1,gbr_pred1,xgbr_pred1,lgbm_pred1]
final_pred = xgbr_new.predict(S_test)
final_pred1 = np.expm1(final_pred)
sub = pd.DataFrame()

sub['Id'] = test_id

sub['SalePrice'] = final_pred1
sub.to_csv('submission.csv',index = False)