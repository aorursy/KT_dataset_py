import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso

import seaborn as sns

from scipy.stats import skew

from scipy.stats.stats import pearsonr





%config InlineBackend.figure_format = 'retina' #set 'png' here when working on notebook

%matplotlib inline
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],

                      test.loc[:,'MSSubClass':'SaleCondition']), ignore_index=True)
x = all_data.loc[np.logical_not(all_data["LotFrontage"].isnull()), "LotArea"]

y = all_data.loc[np.logical_not(all_data["LotFrontage"].isnull()), "LotFrontage"]

# plt.scatter(x, y)

t = (x <= 25000) & (y <= 150)

p = np.polyfit(x[t], y[t], 1)

all_data.loc[all_data['LotFrontage'].isnull(), 'LotFrontage'] = np.polyval(p, all_data.loc[all_data['LotFrontage'].isnull(), 'LotArea'])
all_data.loc[all_data.Alley.isnull(), 'Alley'] = 'NoAlley'

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

all_data.loc[all_data.Utilities.isnull(), 'Utilities'] = 'AllPub'

all_data.loc[all_data.Exterior1st.isnull(), 'Exterior1st'] = 'VinylSd'

all_data.loc[all_data.Exterior2nd.isnull(), 'Exterior2nd'] = 'VinylSd'

all_data.loc[all_data.Functional.isnull(), 'Functional'] = 'Typ'

all_data.loc[all_data.SaleCondition.isnull(), 'SaleCondition'] = 'Normal'

all_data.loc[all_data.SaleCondition.isnull(), 'SaleType'] = 'WD'

all_data.loc[all_data['PoolQC'].isnull(), 'PoolQC'] = 'NoPool'

all_data.loc[all_data['Fence'].isnull(), 'Fence'] = 'NoFence'

all_data.loc[all_data['MiscFeature'].isnull(), 'MiscFeature'] = 'None'

all_data.loc[all_data['Electrical'].isnull(), 'Electrical'] = 'SBrkr'

# only one is null and it has type Detchd

all_data.loc[all_data['GarageArea'].isnull(), 'GarageArea'] = all_data.loc[all_data['GarageType']=='Detchd', 'GarageArea'].mean()

all_data.loc[all_data['GarageCars'].isnull(), 'GarageCars'] = all_data.loc[all_data['GarageType']=='Detchd', 'GarageCars'].median()


all_data = all_data.replace({'Utilities': {'AllPub': 1, 'NoSeWa': 0, 'NoSewr': 0, 'ELO': 0},

                             'Street': {'Pave': 1, 'Grvl': 0 },

                             'FireplaceQu': {'Ex': 5, 

                                            'Gd': 4, 

                                            'TA': 3, 

                                            'Fa': 2,

                                            'Po': 1,

                                            'NoFireplace': 0 

                                            },

                             'Fence': {'GdPrv': 2, 

                                       'GdWo': 2, 

                                       'MnPrv': 1, 

                                       'MnWw': 1,

                                       'NoFence': 0},

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
newer_dwelling = all_data.MSSubClass.replace({20: 1, 

                                            30: 0, 

                                            40: 0, 

                                            45: 0,

                                            50: 0, 

                                            60: 1,

                                            70: 0,

                                            75: 0,

                                            80: 0,

                                            85: 0,

                                            90: 0,

                                           120: 1,

                                           150: 0,

                                           160: 0,

                                           180: 0,

                                           190: 0})

newer_dwelling.name = 'newer_dwelling'
all_data = all_data.replace({'MSSubClass': {20: 'SubClass_20', 

                                            30: 'SubClass_30', 

                                            40: 'SubClass_40', 

                                            45: 'SubClass_45',

                                            50: 'SubClass_50', 

                                            60: 'SubClass_60',

                                            70: 'SubClass_70',

                                            75: 'SubClass_75',

                                            80: 'SubClass_80',

                                            85: 'SubClass_85',

                                            90: 'SubClass_90',

                                           120: 'SubClass_120',

                                           150: 'SubClass_150',

                                           160: 'SubClass_160',

                                           180: 'SubClass_180',

                                           190: 'SubClass_190'}})
# The idea is good quality should rise price, poor quality - reduce price

overall_poor_qu = all_data.OverallQual.copy()

overall_poor_qu = 5 - overall_poor_qu

overall_poor_qu[overall_poor_qu<0] = 0

overall_poor_qu.name = 'overall_poor_qu'



overall_good_qu = all_data.OverallQual.copy()

overall_good_qu = overall_good_qu - 5

overall_good_qu[overall_good_qu<0] = 0

overall_good_qu.name = 'overall_good_qu'



overall_poor_cond = all_data.OverallCond.copy()

overall_poor_cond = 5 - overall_poor_cond

overall_poor_cond[overall_poor_cond<0] = 0

overall_poor_cond.name = 'overall_poor_cond'



overall_good_cond = all_data.OverallCond.copy()

overall_good_cond = overall_good_cond - 5

overall_good_cond[overall_good_cond<0] = 0

overall_good_cond.name = 'overall_good_cond'



exter_poor_qu = all_data.ExterQual.copy()

exter_poor_qu[exter_poor_qu<3] = 1

exter_poor_qu[exter_poor_qu>=3] = 0

exter_poor_qu.name = 'exter_poor_qu'



exter_good_qu = all_data.ExterQual.copy()

exter_good_qu[exter_good_qu<=3] = 0

exter_good_qu[exter_good_qu>3] = 1

exter_good_qu.name = 'exter_good_qu'



exter_poor_cond = all_data.ExterCond.copy()

exter_poor_cond[exter_poor_cond<3] = 1

exter_poor_cond[exter_poor_cond>=3] = 0

exter_poor_cond.name = 'exter_poor_cond'



exter_good_cond = all_data.ExterCond.copy()

exter_good_cond[exter_good_cond<=3] = 0

exter_good_cond[exter_good_cond>3] = 1

exter_good_cond.name = 'exter_good_cond'



bsmt_poor_cond = all_data.BsmtCond.copy()

bsmt_poor_cond[bsmt_poor_cond<3] = 1

bsmt_poor_cond[bsmt_poor_cond>=3] = 0

bsmt_poor_cond.name = 'bsmt_poor_cond'



bsmt_good_cond = all_data.BsmtCond.copy()

bsmt_good_cond[bsmt_good_cond<=3] = 0

bsmt_good_cond[bsmt_good_cond>3] = 1

bsmt_good_cond.name = 'bsmt_good_cond'



garage_poor_qu = all_data.GarageQual.copy()

garage_poor_qu[garage_poor_qu<3] = 1

garage_poor_qu[garage_poor_qu>=3] = 0

garage_poor_qu.name = 'garage_poor_qu'



garage_good_qu = all_data.GarageQual.copy()

garage_good_qu[garage_good_qu<=3] = 0

garage_good_qu[garage_good_qu>3] = 1

garage_good_qu.name = 'garage_good_qu'



garage_poor_cond = all_data.GarageCond.copy()

garage_poor_cond[garage_poor_cond<3] = 1

garage_poor_cond[garage_poor_cond>=3] = 0

garage_poor_cond.name = 'garage_poor_cond'



garage_good_cond = all_data.GarageCond.copy()

garage_good_cond[garage_good_cond<=3] = 0

garage_good_cond[garage_good_cond>3] = 1

garage_good_cond.name = 'garage_good_cond'



kitchen_poor_qu = all_data.KitchenQual.copy()

kitchen_poor_qu[kitchen_poor_qu<3] = 1

kitchen_poor_qu[kitchen_poor_qu>=3] = 0

kitchen_poor_qu.name = 'kitchen_poor_qu'



kitchen_good_qu = all_data.KitchenQual.copy()

kitchen_good_qu[kitchen_good_qu<=3] = 0

kitchen_good_qu[kitchen_good_qu>3] = 1

kitchen_good_qu.name = 'kitchen_good_qu'



qu_list = pd.concat((overall_poor_qu, overall_good_qu, overall_poor_cond, overall_good_cond, exter_poor_qu,

                     exter_good_qu, exter_poor_cond, exter_good_cond, bsmt_poor_cond, bsmt_good_cond, garage_poor_qu,

                     garage_good_qu, garage_poor_cond, garage_good_cond, kitchen_poor_qu, kitchen_good_qu), axis=1)



bad_heating = all_data.HeatingQC.replace({'Ex': 0, 

                                          'Gd': 0, 

                                          'TA': 0, 

                                          'Fa': 1,

                                          'Po': 1})

bad_heating.name = 'bad_heating'

                                          

MasVnrType_Any = all_data.MasVnrType.replace({'BrkCmn': 1,

                                              'BrkFace': 1,

                                              'CBlock': 1,

                                              'Stone': 1,

                                              'None': 0})

MasVnrType_Any.name = 'MasVnrType_Any'



SaleCondition_PriceDown = all_data.SaleCondition.replace({'Abnorml': 1,

                                                          'Alloca': 1,

                                                          'AdjLand': 1,

                                                          'Family': 1,

                                                          'Normal': 0,

                                                          'Partial': 0})

SaleCondition_PriceDown.name = 'SaleCondition_PriceDown'



Neighborhood_Good = pd.DataFrame(np.zeros((all_data.shape[0],1)), columns=['Neighborhood_Good'])

Neighborhood_Good[all_data.Neighborhood=='NridgHt'] = 1

Neighborhood_Good[all_data.Neighborhood=='Crawfor'] = 1

Neighborhood_Good[all_data.Neighborhood=='StoneBr'] = 1

Neighborhood_Good[all_data.Neighborhood=='Somerst'] = 1

Neighborhood_Good[all_data.Neighborhood=='NoRidge'] = 1
from sklearn.svm import SVC

svm = SVC(C=100)

# price categories

pc = pd.Series(np.zeros(train.shape[0]))

pc[:] = 'pc1'

pc[train.SalePrice >= 150000] = 'pc2'

pc[train.SalePrice >= 220000] = 'pc3'

columns_for_pc = ['Exterior1st', 'Exterior2nd', 'RoofMatl', 'Condition1', 'Condition2', 'BldgType']

X_t = pd.get_dummies(train.loc[:, columns_for_pc], sparse=True)

svm.fit(X_t, pc)

pc_pred = svm.predict(X_t)
p = train.SalePrice/100000

plt.hist(p[pc_pred=='pc1'])

plt.hist(p[pc_pred=='pc2'])

plt.hist(p[pc_pred=='pc3'])
price_category = pd.DataFrame(np.zeros((all_data.shape[0],1)), columns=['pc'])

X_t = pd.get_dummies(all_data.loc[:, columns_for_pc], sparse=True)

pc_pred = svm.predict(X_t)

price_category[pc_pred=='pc2'] = 1

price_category[pc_pred=='pc3'] = 2

price_category = price_category.to_sparse()
# Monthes with the lagest number of deals may be significant

season = all_data.MoSold.replace( {1: 0, 

                                   2: 0, 

                                   3: 0, 

                                   4: 1,

                                   5: 1, 

                                   6: 1,

                                   7: 1,

                                   8: 0,

                                   9: 0,

                                  10: 0,

                                  11: 0,

                                  12: 0})

season.name = 'season'



# Numer month is not significant

all_data = all_data.replace({'MoSold': {1: 'Yan', 

                                        2: 'Feb', 

                                        3: 'Mar', 

                                        4: 'Apr',

                                        5: 'May', 

                                        6: 'Jun',

                                        7: 'Jul',

                                        8: 'Avg',

                                        9: 'Sep',

                                        10: 'Oct',

                                        11: 'Nov',

                                        12: 'Dec'}})
all_data = all_data.replace({'CentralAir': {'Y': 1, 

                                            'N': 0}})

all_data = all_data.replace({'PavedDrive': {'Y': 1, 

                                            'P': 0,

                                            'N': 0}})
reconstruct = pd.DataFrame(np.zeros((all_data.shape[0],1)), columns=['Reconstruct'])

reconstruct[all_data.YrSold < all_data.YearRemodAdd] = 1

reconstruct = reconstruct.to_sparse()



recon_after_buy = pd.DataFrame(np.zeros((all_data.shape[0],1)), columns=['ReconstructAfterBuy'])

recon_after_buy[all_data.YearRemodAdd >= all_data.YrSold] = 1

recon_after_buy = recon_after_buy.to_sparse()



build_eq_buy = pd.DataFrame(np.zeros((all_data.shape[0],1)), columns=['Build.eq.Buy'])

build_eq_buy[all_data.YearBuilt >= all_data.YrSold] = 1

build_eq_buy = build_eq_buy.to_sparse()


all_data.YrSold = 2010 - all_data.YrSold
year_map = pd.concat(pd.Series('YearGroup' + str(i+1), index=range(1871+i*20,1891+i*20)) for i in range(0, 7))

all_data.GarageYrBlt = all_data.GarageYrBlt.map(year_map)

all_data.loc[all_data['GarageYrBlt'].isnull(), 'GarageYrBlt'] = 'NoGarage'
all_data.YearBuilt = all_data.YearBuilt.map(year_map)

all_data.YearRemodAdd = all_data.YearRemodAdd.map(year_map)
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



t = all_data[numeric_feats].quantile(.95)

use_max_scater = t[t == 0].index

use_95_scater = t[t != 0].index

all_data[use_max_scater] = all_data[use_max_scater]/all_data[use_max_scater].max()

all_data[use_95_scater] = all_data[use_95_scater]/all_data[use_95_scater].quantile(.95)
t = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 

     '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 

     'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']



all_data.loc[:, t] = np.log1p(all_data.loc[:, t])
# all classes in sklearn requires numeric data only

# transform categorical variable into binary

X = pd.get_dummies(all_data, sparse=True)

X = X.fillna(0)
X = X.drop('RoofMatl_ClyTile', axis=1) # only one is not zero

X = X.drop('Condition2_PosN', axis=1) # only two is not zero

X = X.drop('MSZoning_C (all)', axis=1)

X = X.drop('MSSubClass_SubClass_160', axis=1)

# this features definitely couse overfitting
# add new features

X = pd.concat((X, newer_dwelling, season, reconstruct, recon_after_buy,

               qu_list, bad_heating, MasVnrType_Any, price_category, build_eq_buy), axis=1)
from itertools import product, chain



def poly(X):

    areas = ['LotArea', 'TotalBsmtSF', 'GrLivArea', 'GarageArea', 'BsmtUnfSF']

    # t = [s for s in X.axes[1].get_values() if s not in areas]

    t = chain(qu_list.axes[1].get_values(), 

              ['OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'BsmtCond', 'GarageQual', 'GarageCond',

               'KitchenQual', 'HeatingQC', 'bad_heating', 'MasVnrType_Any', 'SaleCondition_PriceDown', 'Reconstruct',

               'ReconstructAfterBuy', 'Build.eq.Buy'])

    for a, t in product(areas, t):

        x = X.loc[:, [a, t]].prod(1)

        x.name = a + '_' + t

        yield x



XP = pd.concat(poly(X), axis=1)

X = pd.concat((X, XP), axis=1)
X_train = X[:train.shape[0]]

X_test = X[train.shape[0]:]
# the model has become really big

X_train.shape
y = np.log1p(train.SalePrice)
# this come from iterational model improvment. I was trying to understand why the model gives to the two points much better price

x_plot = X_train.loc[X_train['SaleCondition_Partial']==1, 'GrLivArea']

y_plot = y[X_train['SaleCondition_Partial']==1]

plt.scatter(x_plot, y_plot)
outliers_id = np.array([524, 1299])



outliers_id = outliers_id - 1 # id starts with 1, index starts with 0

X_train = X_train.drop(outliers_id)

y = y.drop(outliers_id)

# There are difinetly more outliers
from sklearn.linear_model import Ridge, RidgeCV, LassoCV, LassoLarsCV

from sklearn.model_selection import cross_val_score



def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))

    return(rmse)
alphas_ridge = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]

cv_rmse_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 

            for alpha in alphas_ridge]
print (cv_rmse_ridge)
cv_ridge = pd.Series(cv_rmse_ridge, index = alphas_ridge)

cv_ridge.plot(title = "Validation Ridge")

plt.xlabel("alphas")

plt.ylabel("rmse")
model_ridge = Ridge(alpha = 10).fit(X_train, y)
cv_ridge.min()
#let's look at the residuals as well:

matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)



preds_ridge = pd.DataFrame({"preds Ridge":model_ridge.predict(X_train), "true":y})

preds_ridge["residuals"] = preds_ridge["true"] - preds_ridge["preds Ridge"]

preds_ridge.plot(x = "preds Ridge", y = "residuals",kind = "scatter")
model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)
cv_rmse_lasso = rmse_cv(model_lasso).mean()
print (cv_rmse_lasso)
#let's look at the residuals as well:

matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)



preds_lasso= pd.DataFrame({"preds Lasso":model_lasso.predict(X_train), "true":y})

preds_lasso["residuals"] = preds_lasso["true"] - preds_lasso["preds Lasso"]

preds_lasso.plot(x = "preds Lasso", y = "residuals",kind = "scatter")
import xgboost as xgb

dtrain = xgb.DMatrix(X_train, label = y)

dtest = xgb.DMatrix(X_test)



params = {"max_depth":6, "eta":0.1}

model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)
model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()
model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=6, learning_rate=0.1) #the params were tuned using xgb.cv

model_xgb.fit(X_train, y)
lasso_preds = np.expm1(model_lasso.predict(X_test))

ridge_preds = np.expm1(model_ridge.predict(X_test))

xgb_preds = np.expm1(model_xgb.predict(X_test))
preds = 0.65*lasso_preds + 0.15*ridge_preds + 0.2*xgb_preds
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

import statsmodels.formula.api as sm

from matplotlib import cm
predictions = pd.DataFrame({"lasso":lasso_preds, "ridge":ridge_preds, "xgb": xgb_preds})

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')



x_surf = np.arange(0, 350, 20)                # generate a mesh

y_surf = np.arange(0, 60, 4)

x_surf, y_surf = np.meshgrid(x_surf, y_surf)



ax.plot_surface(lasso_preds,ridge_preds,

                xgb_preds.reshape(lasso_preds.shape),

                rstride=1,

                cstride=1,

                color='None',

                alpha = 0.4)



ax.scatter(predictions['lasso'], predictions['ridge'], predictions['xgb'],

           c='red',

           marker='.',

           alpha=1)





ax.set_xlabel('lasso')

ax.set_ylabel('ridge')

ax.set_zlabel('xgb')
score = [0.12087,0.14608,0.12214,0.12656,0.12097,0.12016,0.12578,0.13411,0.12095,0.11996,0.12144,0.12012,0.11977,0.12042,0.12,0.12066,0.12068,0.16514,0.12099,0.12159,0.12095,0.12081,0.12088,0.11977,0.11968,0.11979,0.11995,0.1216,0.12097,0.11766,0.11948,0.11543]



plt.title("Our score")

plt.plot(range(0,len(score)), score)

plt.xlabel('Submission')

plt.ylabel('Score')

plt.show()
solution = pd.DataFrame({"id":test.Id, "SalePrice":preds})

solution.to_csv("House_Price_final_12032017.csv", index = False)
