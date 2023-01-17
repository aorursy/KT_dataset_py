# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from sklearn import metrics

from scipy.stats import skew

from sklearn.linear_model import LinearRegression

from sklearn.cross_validation import cross_val_score



train = pd.read_csv('../input/train.csv', low_memory=False, dtype = {'SalePrice': float})

test = pd.read_csv('../input/test.csv', low_memory=False)

train=train.drop(train.loc[:,['Condition2','Utilities','RoofMatl','Heating','PoolQC','Fence','MiscFeature','Street','Alley']],1)

test=test.drop(test.loc[:,['Condition2','Utilities','RoofMatl','Heating','PoolQC','Fence','MiscFeature','Street','Alley']],1)



all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],

                      test.loc[:,'MSSubClass':'SaleCondition']))

#all_data.loc[all_data.Alley.isnull(), 'Alley'] = 'NoAlley'

all_data.loc[all_data.MasVnrType.isnull(), 'MasVnrType'] = 'None' # no good

all_data.loc[all_data.MasVnrType == 'None', 'MasVnrArea'] = 0

all_data.loc[all_data.BsmtQual.isnull(), 'BsmtQual'] = 'NoBsmt'

all_data.loc[all_data.BsmtCond.isnull(), 'BsmtCond'] = 'NoBsmt'

all_data.loc[all_data.BsmtExposure.isnull(), 'BsmtExposure'] = 'NoBsmt'

all_data.loc[all_data.BsmtFinType1.isnull(), 'BsmtFinType1'] = 'NoBsmt'

all_data.loc[all_data.BsmtFinType2.isnull(), 'BsmtFinType2'] = 'NoBsmt'

all_data.loc[all_data.BsmtFinType1=='NoBsmt', 'BsmtFinSF1'] = 0

all_data.loc[all_data.BsmtFinType2=='NoBsmt', 'BsmtFinSF2'] = 0

all_data.loc[all_data.BsmtFinSF1.isnull(), 'BsmtFinSF1'] = all_data.BsmtFinSF1.median()

all_data.loc[all_data.BsmtQual=='NoBsmt', 'BsmtUnfSF'] = 0

all_data.loc[all_data.BsmtUnfSF.isnull(), 'BsmtUnfSF'] = all_data.BsmtUnfSF.median()

all_data.loc[all_data.BsmtQual=='NoBsmt', 'TotalBsmtSF'] = 0

all_data.loc[all_data.FireplaceQu.isnull(), 'FireplaceQu'] = 'NoFireplace'

all_data.loc[all_data.GarageType.isnull(), 'GarageType'] = 'NoGarage'

all_data.loc[all_data.GarageFinish.isnull(), 'GarageFinish'] = 'NoGarage'

all_data.loc[all_data.GarageQual.isnull(), 'GarageQual'] = 'NoGarage'

all_data.loc[all_data.GarageCond.isnull(), 'GarageCond'] = 'NoGarage'

all_data.loc[all_data.BsmtFullBath.isnull(), 'BsmtFullBath'] = 0

all_data.loc[all_data.BsmtHalfBath.isnull(), 'BsmtHalfBath'] = 0

all_data.loc[all_data.KitchenQual.isnull(), 'KitchenQual'] = 'TA'

all_data.loc[all_data.MSZoning.isnull(), 'MSZoning'] = 'RL'

#all_data.loc[all_data.Utilities.isnull(), 'Utilities'] = 'AllPub'

all_data.loc[all_data.Exterior1st.isnull(), 'Exterior1st'] = 'VinylSd'

all_data.loc[all_data.Exterior2nd.isnull(), 'Exterior2nd'] = 'VinylSd'

all_data.loc[all_data.Functional.isnull(), 'Functional'] = 'Typ'

all_data.loc[all_data.SaleCondition.isnull(), 'SaleCondition'] = 'Normal'

all_data.loc[all_data.SaleCondition.isnull(), 'SaleType'] = 'WD'

#all_data.loc[all_data['PoolQC'].isnull(), 'PoolQC'] = 'NoPool'

#all_data.loc[all_data['Fence'].isnull(), 'Fence'] = 'NoFence'

#all_data.loc[all_data['MiscFeature'].isnull(), 'MiscFeature'] = 'None'

all_data.loc[all_data['Electrical'].isnull(), 'Electrical'] = 'SBrkr'

# only one is null and it has type Detchd

all_data.loc[all_data['GarageArea'].isnull(), 'GarageArea'] = all_data.loc[all_data['GarageType']=='Detchd', 'GarageArea'].mean()

all_data.loc[all_data['GarageCars'].isnull(), 'GarageCars'] = all_data.loc[all_data['GarageType']=='Detchd', 'GarageCars'].median()

all_data = all_data.replace({'Street': {'Pave': 1, 'Grvl': 0 },

                             'FireplaceQu': {'Ex': 5, 

                                            'Gd': 4, 

                                            'TA': 3, 

                                            'Fa': 2,

                                            'Po': 1,

                                            'NoFireplace': 0 

                                            },

                             

                             'ExterQual': {'Ex': 5, 

                                            'Gd': 4, 

                                            'TA': 3, 

                                            'Fa': 2,

                                            'Po': 1

                                            },

                             'ExterCond': {'Ex': 5, 

                                            'Gd': 4, 

                                            'TA': 3, 

                                            'Fa': 2,

                                            'Po': 1

                                            },

                             'BsmtQual': {'Ex': 5, 

                                            'Gd': 4, 

                                            'TA': 3, 

                                            'Fa': 2,

                                            'Po': 1,

                                            'NoBsmt': 0},

                             'BsmtExposure': {'Gd': 3, 

                                            'Av': 2, 

                                            'Mn': 1,

                                            'No': 0,

                                            'NoBsmt': 0},

                             'BsmtCond': {'Ex': 5, 

                                            'Gd': 4, 

                                            'TA': 3, 

                                            'Fa': 2,

                                            'Po': 1,

                                            'NoBsmt': 0},

                             'GarageQual': {'Ex': 5, 

                                            'Gd': 4, 

                                            'TA': 3, 

                                            'Fa': 2,

                                            'Po': 1,

                                            'NoGarage': 0},

                             'GarageCond': {'Ex': 5, 

                                            'Gd': 4, 

                                            'TA': 3, 

                                            'Fa': 2,

                                            'Po': 1,

                                            'NoGarage': 0},

                             'KitchenQual': {'Ex': 5, 

                                            'Gd': 4, 

                                            'TA': 3, 

                                            'Fa': 2,

                                            'Po': 1},

                             'Functional': {'Typ': 0,

                                            'Min1': 1,

                                            'Min2': 1,

                                            'Mod': 2,

                                            'Maj1': 3,

                                            'Maj2': 4,

                                            'Sev': 5,

                                            'Sal': 6}                             

                            })



train = train.replace({ 'Street': {'Pave': 1, 'Grvl': 0 },

                             'FireplaceQu': {'Ex': 5, 

                                            'Gd': 4, 

                                            'TA': 3, 

                                            'Fa': 2,

                                            'Po': 1,

                                            'NoFireplace': 0 

                                            },

                             

                             'ExterQual': {'Ex': 5, 

                                            'Gd': 4, 

                                            'TA': 3, 

                                            'Fa': 2,

                                            'Po': 1

                                            },

                             'ExterCond': {'Ex': 5, 

                                            'Gd': 4, 

                                            'TA': 3, 

                                            'Fa': 2,

                                            'Po': 1

                                            },

                             'BsmtQual': {'Ex': 5, 

                                            'Gd': 4, 

                                            'TA': 3, 

                                            'Fa': 2,

                                            'Po': 1,

                                            'NoBsmt': 0},

                             'BsmtExposure': {'Gd': 3, 

                                            'Av': 2, 

                                            'Mn': 1,

                                            'No': 0,

                                            'NoBsmt': 0},

                             'BsmtCond': {'Ex': 5, 

                                            'Gd': 4, 

                                            'TA': 3, 

                                            'Fa': 2,

                                            'Po': 1,

                                            'NoBsmt': 0},

                             'GarageQual': {'Ex': 5, 

                                            'Gd': 4, 

                                            'TA': 3, 

                                            'Fa': 2,

                                            'Po': 1,

                                            'NoGarage': 0},

                             'GarageCond': {'Ex': 5, 

                                            'Gd': 4, 

                                            'TA': 3, 

                                            'Fa': 2,

                                            'Po': 1,

                                            'NoGarage': 0},

                             'KitchenQual': {'Ex': 5, 

                                            'Gd': 4, 

                                            'TA': 3, 

                                            'Fa': 2,

                                            'Po': 1},

                             'Functional': {'Typ': 0,

                                            'Min1': 1,

                                            'Min2': 1,

                                            'Mod': 2,

                                            'Maj1': 3,

                                            'Maj2': 4,

                                            'Sev': 5,

                                            'Sal': 6}                             

                            })



all_data = all_data.replace({'CentralAir': {'Y': 1, 

                                            'N': 0}})

all_data = all_data.replace({'PavedDrive': {'Y': 1, 

                                            'P': 0,

                                            'N': 0}})

train= train.replace({'CentralAir': {'Y': 1, 

                                            'N': 0}})

train = train.replace({'PavedDrive': {'Y': 1, 

                                            'P': 0,

                                            'N': 0}})

#all_data.add('Age', axis=1, level=None, fill_value=None)

all_data['YearRemodAdd']=2016-all_data['YearRemodAdd']

train['YearRemodAdd']=2016-train['YearRemodAdd']



#all_data.drop('YearRemodAdd',axis=1,inplace=True)

#train.drop('YearRemodAdd',axis=1,inplace=True)

#all_data=all_data[all_data.isnull().any(axis=1)]

for c in all_data:

    if sum(all_data[c].isnull()) >= 600:

        all_data.drop(c, axis=1, inplace=True)

#log transform the target:

train["SalePrice"] = np.log1p(train["SalePrice"])



#log transform skewed numeric features:

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

skewed_feats = skewed_feats[skewed_feats > 0.65]

skewed_feats = skewed_feats.index



all_data[skewed_feats] = np.log1p(all_data[skewed_feats])





all_data = pd.get_dummies(all_data)



#filling NA's with the mean of the column:

all_data = all_data.fillna( 1 )





#creating matrices for sklearn:

X_train = all_data[:train.shape[0]]

X_test = all_data[train.shape[0]:]

y = train.SalePrice



print (X_train.shape, X_test.shape, y.shape)

#print (all_data['YearRemodAdd'].head)



def rmse_cv(model):

    rmse = np.sqrt(-cross_val_score(model, X_train, y, scoring="mean_squared_error", cv = 5))

    return(rmse)
def runAlgo(algo, x, y, X_test):

    algo.fit(x, y)

    y_pred = algo.predict(X_test)

    print (np.sqrt(metrics.mean_squared_error(y, y_pred)))
#from sklearn.ensemble import RandomForestRegressor

# Algo 3

#rfr = RandomForestRegressor(n_estimators=800)

#rfr = RandomForestRegressor()

#rfr.fit(X_train, y)

#y_pred = np.expm1(rfr.predict(X_train))

#print (np.sqrt(metrics.mean_squared_error(y, y_pred)))



# Test data

#y_pred_ = np.expm1(rfr.predict(X_test))

#solution = pd.DataFrame({"Id":test.Id, "SalePrice":y_pred_})

#solution.to_csv("kaggle.csv", index = False)
#from sklearn.svm import SVR

#clf = SVR()

#clf.fit(X_train, y) 

#SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.2, gamma='auto',

#    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)

#NuSVR(C=1.0, cache_size=200, coef0=0.0, degree=3, gamma='auto',

     # kernel='rbf', max_iter=-1, nu=0.1, shrinking=True, tol=0.001,

     # verbose=False)



#y_pred_ = np.expm1(clf.predict(X_test))

#solution = pd.DataFrame({"Id":test.Id, "SalePrice":y_pred_})

#solution.to_csv("kaggle.csv", index = False)
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV

from sklearn.model_selection import cross_val_score

import xgboost as xgb

def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))

    return(rmse)

#model_lasso = LassoCV(alphas = [1.2, 0.25, 0.028, 0.0040,0.0004366]).fit(X_train, y)

model_lasso = LassoCV(alphas=[1.2, 0.25, 0.028, 0.0040,0.0004366], max_iter=80000).fit(X_train, y)

lpreds = np.expm1(model_lasso.predict(X_test))



dtrain = xgb.DMatrix(X_train, label = y)

dtest = xgb.DMatrix(X_test)



params = {"max_depth":2, "eta":0.1}

model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)



model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv

model_xgb.fit(X_train, y)

xgb_preds = np.expm1(model_xgb.predict(X_test))

preds = 0.73*lpreds + 0.27*xgb_preds

solution = pd.DataFrame({"Id":test.Id, "SalePrice":preds})

solution.to_csv("kaggle.csv", index = False)