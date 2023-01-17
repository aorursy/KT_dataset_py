## Abstract ##
import pandas as pd

import numpy as np



import seaborn as sns

import matplotlib



import matplotlib.pyplot as plt

from scipy.stats import skew

from scipy.stats.stats import pearsonr





%config InlineBackend.figure_format = 'retina' #set 'png' here when working on notebook

%matplotlib inline
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],

                      test.loc[:,'MSSubClass':'SaleCondition']))
# I have no idea how to do it better. Probably, it is better to do nothing

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
# where we have order we will use numeric

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
train['SalePrice'].describe()
train['SalePrice'].describe()
#skewness and kurtosis

print("Skewness: %f" % train['SalePrice'].skew())

print("Kurtosis: %f" % train['SalePrice'].kurt())
#scatter plot grlivarea/saleprice

var = 'GrLivArea'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


sns.distplot(train['SalePrice']);
train.head()
#box plot overallqual/saleprice

var = 'OverallQual'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)

prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})

prices.hist()
all_data = pd.get_dummies(all_data)
#filling NA's with the mean of the column:

all_data = all_data.fillna(all_data.mean())
#creating matrices for sklearn:

X_train = all_data[:train.shape[0]]

X_test = all_data[train.shape[0]:]

y = train.SalePrice


from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV

from sklearn.model_selection import cross_val_score



def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))

    return(rmse)
## New models ##
import xgboost as xgb


dtrain = xgb.DMatrix(X_train, label = y)

dtest = xgb.DMatrix(X_test)



params = {"max_depth":2, "eta":0.1}

model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)
model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()
model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv

model_xgb.fit(X_train, y)
corr = train.select_dtypes(include = ['float64', 'int64']).iloc[:, 1:].corr()

plt.figure(figsize=(12, 12))

sns.heatmap(corr, vmax=1, square=True)
cor_dict = corr['SalePrice'].to_dict()

del cor_dict['SalePrice']

print("List the numerical features decendingly by their correlation with Sale Price:\n")

for ele in sorted(cor_dict.items(), key = lambda x: -abs(x[1])):

    print("{0}: \t{1}".format(*ele))
#saleprice correlation matrix

k = 10 #number of variables for heatmap

cols = corr.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
