# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.info()
train['SalePrice'].describe()
sns.distplot(train['SalePrice'])
train['YearBuilt'].describe()
sns.distplot(train['LotArea'])
print(train['SalePrice'].skew())
print(train['SalePrice'].kurt())
print(train['LotArea'].skew())
print(train['LotArea'].kurt())
train.plot.scatter(title='Living Area Sq. Ft. VS. Sale Price', x='GrLivArea', y='SalePrice', ylim=(0,800000))
train.plot.scatter(title='Year Built VS. Sale Price', x='YearBuilt', y='SalePrice', ylim=(0,800000))
plt.figure(figsize=(10,10))
plt.title('Distribution of Sale Price for House Style')

sns.boxplot(x='HouseStyle', y='SalePrice', data=train)
plt.figure(figsize=(20,20))
plt.title('Distribution of Sale Price for Neighborhood')

sns.boxplot(x='Neighborhood', y='SalePrice', data=train)
plt.subplots(figsize = (15,15))
sns.heatmap(train.corr(),
           cmap = 'RdBu_r',
           #linewidths=0.1,
           #linecolor='white',
           vmax=0.9,
           square=True)
corrmat = train.corr()
cols = corrmat.nlargest(10, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm,
                cbar=True,
                annot=True,
                square=True,
                fmt='.2f',
                annot_kws={'size': 10},
                yticklabels=cols.values,
                xticklabels=cols.values)
plt.show()
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[cols], size = 3)
plt.show()
total = train.isnull().sum().sort_values(ascending = False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing = pd.concat([total,percent], axis=1, keys=['Total', 'Percent'])
missing.head(20)
train = train.drop((missing[missing['Total'] > 1]).index, 1)
train = train.drop(train.loc[train['Electrical'].isnull()].index)
train.isnull().sum().max()
from sklearn.preprocessing import StandardScaler

scaled_price = StandardScaler().fit_transform(train['SalePrice'][:, np.newaxis])
low = scaled_price[scaled_price[:,0].argsort()][:10]
high = scaled_price[scaled_price[:,0].argsort()][-10:]

print(low)
print()
print(high)
var = 'GrLivArea'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
train.sort_values(by='GrLivArea', ascending=False)[:2]
train = train.drop(train[train['Id'] == 1299].index)
train = train.drop(train[train['Id'] == 524 ].index)
var = 'GrLivArea'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
var = 'YearBuilt'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
train = train.drop(train[(train['YearBuilt'] < 1990) & (train['SalePrice'] > 300000)].index)
train = train.drop(train[(train['YearBuilt'] > 1980) & (train['SalePrice'] > 700000)].index)
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
from scipy.stats import norm
from scipy import stats

sns.distplot(train['SalePrice'], fit=norm)
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
train['SalePrice'] = np.log(train['SalePrice'])
sns.distplot(train['SalePrice'], fit=norm)
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
sns.distplot(train['LotArea'], fit=norm)
fig = plt.figure()
res = stats.probplot(train['LotArea'], plot=plt)
train['LotArea'] = np.log(train['LotArea'])
sns.distplot(train['LotArea'], fit=norm)
fig = plt.figure()
res = stats.probplot(train['LotArea'], plot=plt)
sns.distplot(train['PoolArea'], fit=norm)
fig = plt.figure()
res = stats.probplot(train['PoolArea'], plot=plt)
train['HasPool'] = pd.Series(len(train['PoolArea']), index=train.index)
train['HasPool'] = 0
train.loc[train['PoolArea']>0,'HasPool'] = 1
train.loc[train['HasPool']==1,'PoolArea'] = np.log(train['PoolArea'])
sns.distplot(train[train['PoolArea']>0]['PoolArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(train[train['PoolArea']>0]['PoolArea'], plot=plt)
train = pd.read_csv("../input/train.csv")
diff_names = ['RoofMatl_Roll', 'Condition2_RRAe', 'HouseStyle_2.5Fin', 'Electrical_Mix', 'Heating_Floor', 
              'RoofMatl_Membran', 'Exterior2nd_Other', 'Heating_OthW', 'MiscFeature_TenC', 'Condition2_RRAn', 
              'Utilities_NoSeWa', 'Condition2_RRNn', 'Exterior1st_ImStucc', 'Exterior1st_Stone', 'RoofMatl_Metal']
train = train[train['GrLivArea'] < 4000]
train.columns[train.isnull().any()]
train.loc[:, 'LotFrontage' ] = train.loc[:, 'LotFrontage' ].fillna(0)
train.loc[:, 'Alley'       ] = train.loc[:, 'Alley'       ].fillna("none")
train.loc[:, 'MasVnrType'  ] = train.loc[:, 'MasVnrType'  ].fillna('none')
train.loc[:, 'MasVnrArea'  ] = train.loc[:, 'MasVnrArea'  ].fillna(0)
train.loc[:, 'BsmtQual'    ] = train.loc[:, 'BsmtQual'    ].fillna("no")
train.loc[:, 'BsmtCond'    ] = train.loc[:, 'BsmtCond'    ].fillna("no")
train.loc[:, 'BsmtExposure'] = train.loc[:, 'BsmtExposure'].fillna("no")
train.loc[:, 'BsmtFinType1'] = train.loc[:, 'BsmtFinType1'].fillna("no")
train.loc[:, 'BsmtFinType2'] = train.loc[:, 'BsmtFinType2'].fillna("no")
train.loc[:, 'FireplaceQu' ] = train.loc[:, 'FireplaceQu' ].fillna("no")
train.loc[:, 'PoolQC'      ] = train.loc[:, 'PoolQC'      ].fillna("no")
train.loc[:, 'Fence'       ] = train.loc[:, 'Fence'       ].fillna("no")
train.loc[:, 'MiscFeature' ] = train.loc[:, 'MiscFeature' ].fillna("none")

train = train.drop(train.loc[train['Electrical' ].isnull()].index)
train = train.drop(train.loc[train['GarageYrBlt'].isnull()].index)
train.SalePrice = np.log1p(train.SalePrice)
y = train.SalePrice
train.columns[train.isnull().any()]
train = train.replace({'Street'       : { 'Grvl' : 1, 'Pave' : 2 },
                       'Alley'        : { 'none' : 0, 'Grvl' : 1, 'Pave' : 2},
                       'ExterQual'    : { 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5 },
                       'ExterCond'    : { 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5 },
                       'BsmtQual'     : { 'no' : 0, 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5 },
                       'BsmtCond'     : { 'no' : 0, 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5 },
                       'BsmtExposure' : { 'no' : 0, 'No' : 1, 'Mn' : 2, 'Av' : 3, 'Gd' : 4 },
                       'BsmtFinType1' : { 'no' : 0, 'Unf' : 1, 'LwQ' : 2, 'Rec' : 3, 'BLQ' : 4, 'ALQ' : 5, 'GLQ' : 6 },
                       'BsmtFinType2' : { 'no' : 0, 'Unf' : 1, 'LwQ' : 2, 'Rec' : 3, 'BLQ' : 4, 'ALQ' : 5, 'GLQ' : 6 },
                       'HeatingQC'    : { 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5 },
                       'CentralAir'   : { 'N' : 0, 'Y' : 1 },
                       'KitchenQual'  : { 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5 },
                       'Functional'   : { 'Sal' : 1, 'Sev' : 2, 'Maj2' : 3, 'Maj1' : 4, 'Mod' : 5, 'Min2' : 6, 'Min1' : 7, 'Typ' : 8 },
                       'FireplaceQu'  : { 'no' : 0, 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5 },
                       'GarageQual'   : { 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5 },
                       'GarageCond'   : { 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5 },
                       'PavedDrive'   : { 'N' : 1, 'P': 2, 'Y' : 3},
                       'PoolQC'       : { 'no' : 0, 'Fa' : 1, 'TA' : 2, 'Gd' : 3, 'Ex' : 4 },
                       'Fence'        : { 'no' : 0, 'MnWw' : 1, 'GdWo' : 2, 'MnPrv' : 3, 'GdPrv' : 4 }
                      })
categorical_data =  train.select_dtypes(include=['object']).dtypes
train['SimpleOverallQual']   = train.OverallQual .replace({ 1 : 1, 2 : 1, 3: 1, 4 : 2, 5 : 2, 6 : 2, 7 : 3, 
                                                           8 : 3, 9 : 3, 10 : 3 })
train['SimpleOverallCond']   = train.OverallCond .replace({ 1 : 1, 2 : 1, 3: 1, 4 : 2, 5 : 2, 6 : 2, 7 : 3, 
                                                           8 : 3, 9 : 3, 10 : 3 })
train['SimpleExterQual' ]   = train.ExterQual   .replace({ 1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2 })
train['SimpleExterCond' ]   = train.ExterCond   .replace({ 1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2 })
train['SimpleBsmtQual'  ]   = train.BsmtQual    .replace({ 0 : 0, 1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2 })
train['SimpleBsmtCond'  ]   = train.BsmtCond    .replace({ 0 : 0, 1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2 })
train['SimpleBsmtExposure'] = train.BsmtExposure.replace({ 0 : 0, 1 : 1, 2 : 1, 3 : 2, 4 : 2 })
train['SimpleBsmtFinType1'] = train.BsmtFinType1.replace({ 0 : 0, 1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2, 6 : 2 })
train['SimpleBsmtFinType2'] = train.BsmtFinType2.replace({ 0 : 0, 1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2, 6 : 2 })
train['SimpleHeatingQC'   ] = train.HeatingQC   .replace({ 1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2 })
train['SimpleKitchenQual' ] = train.KitchenQual .replace({ 1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2 })
train['SimpleFunctional'  ] = train.Functional  .replace({ 1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2 })
train['SimpleFireplaceQu' ] = train.FireplaceQu .replace({ 0 : 0, 1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2 })
train['SimpleGarageQual'  ] = train.GarageQual  .replace({ 1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2 })
train['SimpleGarageCond'  ] = train.GarageCond  .replace({ 1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2 })
train['SimplePoolQC'      ] = train.PoolQC      .replace({ 0 : 0, 1 : 1, 2 : 1, 3 : 2, 4 : 2 })
train['SimpleFence'       ] = train.Fence       .replace({ 0 : 0, 1 : 1, 2 : 1, 3 : 2, 4 : 2 })
train['OverallScore']   = train['SimpleOverallQual'] * train['SimpleOverallCond']
train['ExterScore'  ]   = train['SimpleExterCond'] * train['SimpleExterQual']
train['BsmtScore'   ]   = train['SimpleBsmtCond'] * train['SimpleBsmtExposure'] * train['SimpleBsmtFinType1'] * train['SimpleBsmtFinType2']
train['GarageScore' ]   = train['SimpleGarageCond'] * train['SimpleGarageQual']
train['KitchenScore']   = train['SimpleKitchenQual'] * train['KitchenAbvGr']
train['FireplaceScore'] = train['FireplaceQu'] * train['Fireplaces']
train['PoolScore'] = train['PoolArea'] * train['SimplePoolQC']

train['OverallHouseSF'] = train['GrLivArea'] + train['TotalBsmtSF'] 
train['PorchSF'] = train['OpenPorchSF'] + train['EnclosedPorch'] + train['3SsnPorch'] \
+ train['ScreenPorch']
train["TotalBath"] = train["BsmtFullBath"] + (0.5 * train["BsmtHalfBath"]) + \
train["FullBath"] + (0.5 * train["HalfBath"])



train["HasMasVnr"] = train.MasVnrType.replace({"BrkCmn" : 1, "BrkFace" : 1, "CBlock" : 1, 
                                               "Stone" : 1, "None" : 0})

train["BoughtOffPlan"] = train.SaleCondition.replace({"Abnorml" : 0, "Alloca" : 0, "AdjLand" : 0, 
                                                      "Family" : 0, "Normal" : 0, "Partial" : 1})
train['SquareOverallHouseSF'   ] = train['OverallHouseSF'   ]**2 
train['SquareOverallQual'      ] = train['OverallQual'      ]**2 
train['SquareGrLivArea'        ] = train['GrLivArea'        ]**2 
train['SquareSimpleOverallQual'] = train['SimpleOverallQual']**2 
train['SquareExterQual'        ] = train['ExterQual'        ]**2 
train['SquareTotalBath'        ] = train['TotalBath'        ]**2 
train['SquareKitchenQual'      ] = train['KitchenQual'      ]**2 
train['SquareGarageCars'       ] = train['GarageCars'       ]**2 
train['SquareSimpleExterQual'  ] = train['SimpleExterQual'  ]**2 

train['CubedOverallHouseSF'   ] = train['OverallHouseSF'    ]**3
train['CubedOverallQual'      ] = train['OverallQual'       ]**3
train['CubedGrLivArea'        ] = train['GrLivArea'         ]**3
train['CubedSimpleOverallQual'] = train['SimpleOverallQual' ]**3
train['CubedExterQual'        ] = train['ExterQual'         ]**3
train['CubedTotalBath'        ] = train['TotalBath'         ]**3
train['CubedKitchenQual'      ] = train['KitchenQual'       ]**3
train['CubedGarageCars'       ] = train['GarageCars'        ]**3
train['CubedSimpleExterQual'  ] = train['SimpleExterQual'   ]**3
train['BsmtExposure'      ] = train['BsmtExposure'      ].apply(lambda x : np.log(x) if x != 0 else 0)
train['BsmtFinSF1'        ] = train['BsmtFinSF1'        ].apply(lambda x : np.log(x) if x != 0 else 0)
train['BsmtFinType2'      ] = train['BsmtFinType2'      ].apply(lambda x : np.log(x) if x != 0 else 0)
train['BsmtFinSF2'        ] = train['BsmtFinSF2'        ].apply(lambda x : np.log(x) if x != 0 else 0)
train['BsmtUnfSF'         ] = train['BsmtUnfSF'         ].apply(lambda x : np.log(x) if x != 0 else 0)
train['TotalBsmtSF'       ] = train['TotalBsmtSF'       ].apply(lambda x : np.log(x) if x != 0 else 0)
train['SimpleBsmtCond'    ] = train['SimpleBsmtCond'    ].apply(lambda x : np.log(x) if x != 0 else 0)
train['SimpleBsmtExposure'] = train['SimpleBsmtExposure'].apply(lambda x : np.log(x) if x != 0 else 0)
train['SimpleBsmtFinType2'] = train['SimpleBsmtFinType2'].apply(lambda x : np.log(x) if x != 0 else 0)
train['BsmtScore'         ] = train['BsmtScore'         ].apply(lambda x : np.log(x) if x != 0 else 0)
train['2ndFlrSF'          ] = train['2ndFlrSF'          ].apply(lambda x : np.log(x) if x != 0 else 0)
train['HalfBath'          ] = train['HalfBath'          ].apply(lambda x : np.log(x) if x != 0 else 0)
train['Alley'             ] = train['Alley'             ].apply(lambda x : np.log(x) if x != 0 else 0)
train['MasVnrArea'        ] = train['MasVnrArea'        ].apply(lambda x : np.log(x) if x != 0 else 0)
train['LowQualFinSF'      ] = train['LowQualFinSF'      ].apply(lambda x : np.log(x) if x != 0 else 0)
train['Fireplaces'        ] = train['Fireplaces'        ].apply(lambda x : np.log(x) if x != 0 else 0)
train['FireplaceScore'    ] = train['FireplaceScore'    ].apply(lambda x : np.log(x) if x != 0 else 0)
train['PoolArea'          ] = train['PoolArea'          ].apply(lambda x : np.log(x) if x != 0 else 0)
train['PoolQC'            ] = train['PoolQC'            ].apply(lambda x : np.log(x) if x != 0 else 0)
train['SimplePoolQC'      ] = train['SimplePoolQC'      ].apply(lambda x : np.log(x) if x != 0 else 0)
train['PoolScore'         ] = train['PoolScore'         ].apply(lambda x : np.log(x) if x != 0 else 0)
train['Fence'             ] = train['Fence'             ].apply(lambda x : np.log(x) if x != 0 else 0)
train['SimpleFence'       ] = train['SimpleFence'       ].apply(lambda x : np.log(x) if x != 0 else 0)
train['MiscVal'           ] = train['MiscVal'           ].apply(lambda x : np.log(x) if x != 0 else 0)
train['PorchSF'           ] = train['PorchSF'           ].apply(lambda x : np.log(x) if x != 0 else 0)
from scipy.stats import norm
from scipy import stats
sns.distplot(train[train['PorchSF']>0]['PorchSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(train[train['PorchSF']>0]['PorchSF'], plot=plt)
corrmat = train.corr()
cols = corrmat.nlargest(10, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm,
                cbar=True,
                annot=True,
                square=True,
                fmt='.2f',
                annot_kws={'size': 10},
                yticklabels=cols.values,
                xticklabels=cols.values)
plt.show()
categorical_features = train.select_dtypes(include = ["object"]).columns
numerical_features = train.select_dtypes(exclude = ["object"]).columns
numerical_features = numerical_features.drop("SalePrice")
train_num = train[numerical_features]
train_cat = train[categorical_features]

train_cat = pd.get_dummies(train_cat)
train = pd.concat([train_num, train_cat], axis = 1)

train_names = train.columns.get_values().tolist()
train.shape
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, make_scorer
from math import sqrt
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size = 0.3, random_state = 0)
stdSc = StandardScaler()
X_train.loc[:, numerical_features] = stdSc.fit_transform(X_train.loc[:, numerical_features])
X_test.loc[:, numerical_features] = stdSc.transform(X_test.loc[:, numerical_features])
lr = LinearRegression()
lr.fit(X_train, y_train)

y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

plt.scatter(y_train_pred, y_train_pred - y_train, c = "blue", marker = "s", label = "Training data")
plt.scatter(y_test_pred, y_test_pred - y_test, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("Linear regression")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.legend(loc = "upper left")
plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")
plt.show()

plt.scatter(y_train_pred, y_train, c = "blue", marker = "s", label = "Training data")
plt.scatter(y_test_pred, y_test, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("Linear regression")
plt.xlabel("Predicted values")
plt.ylabel("Real values")
plt.legend(loc = "upper left")
plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")
plt.show()
rmse_test = round(sqrt(mean_squared_error(y_test, y_test_pred)), 3)
print('Root Mean Square Error of Test Data is: ', rmse_test)

rmse_train = round(sqrt(mean_squared_error(y_train, y_train_pred)), 3)
print('Root Mean Square Error of Train Data is: ', rmse_train)
lasso = LassoCV(alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 
                          0.3, 0.6, 1], 
                max_iter = 50000, cv = 10)
lasso.fit(X_train, y_train)
alpha = lasso.alpha_

lasso = LassoCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, 
                          alpha * .85, alpha * .9, alpha * .95, alpha, alpha * 1.05, 
                          alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, alpha * 1.35, 
                          alpha * 1.4], 
                max_iter = 50000, cv = 10)
lasso.fit(X_train, y_train)
alpha = lasso.alpha_

y_train_las = lasso.predict(X_train)
y_test_las = lasso.predict(X_test)

plt.scatter(y_train_las, y_train_las - y_train, c = "blue", marker = "s", label = "Training data")
plt.scatter(y_test_las, y_test_las - y_test, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("Linear regression with Lasso regularization")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.legend(loc = "upper left")
plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")
plt.show()

plt.scatter(y_train_las, y_train, c = "blue", marker = "s", label = "Training data")
plt.scatter(y_test_las, y_test, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("Linear regression with Lasso regularization")
plt.xlabel("Predicted values")
plt.ylabel("Real values")
plt.legend(loc = "upper left")
plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")
plt.show()

coefs = pd.Series(lasso.coef_, index = X_train.columns)
print("Lasso picked " + str(sum(coefs != 0)) + " features and eliminated the other " +  \
      str(sum(coefs == 0)) + " features")
imp_coefs = pd.concat([coefs.sort_values().head(10),
                     coefs.sort_values().tail(10)])
imp_coefs.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")
plt.show()
rmse_test = round(sqrt(mean_squared_error(y_test, y_test_las)), 3)
print('Root Mean Square Error of Test Data is: ', rmse_test)

rmse_train = round(sqrt(mean_squared_error(y_train, y_train_las)), 3)
print('Root Mean Square Error of Train Data is: ', rmse_train)
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=0)
rf.fit(X_train, y_train)
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, pearsonr
predicted_train = rf.predict(X_train)
predicted_test = rf.predict(X_test)
test_score = r2_score(y_test, predicted_test)
spearman = spearmanr(y_test, predicted_test)
pearson = pearsonr(y_test, predicted_test)
print(f'Out-of-bag R-2 score estimate: {rf.oob_score_:>5.3}')
print(f'Test data R-2 score: {test_score:>5.3}')
print(f'Test data Spearman correlation: {spearman[0]:.3}')
print(f'Test data Pearson correlation: {pearson[0]:.3}')
test = pd.read_csv("../input/test.csv")
test.loc[:, 'LotFrontage' ] = test.loc[:, 'LotFrontage' ].fillna(0)
test.loc[:, 'Alley'       ] = test.loc[:, 'Alley'       ].fillna("none")
test.loc[:, 'MasVnrType'  ] = test.loc[:, 'MasVnrType'  ].fillna('none')
test.loc[:, 'MasVnrArea'  ] = test.loc[:, 'MasVnrArea'  ].fillna(0)
test.loc[:, 'BsmtQual'    ] = test.loc[:, 'BsmtQual'    ].fillna("no")
test.loc[:, 'BsmtCond'    ] = test.loc[:, 'BsmtCond'    ].fillna("no")
test.loc[:, 'BsmtExposure'] = test.loc[:, 'BsmtExposure'].fillna("no")
test.loc[:, 'BsmtFinType1'] = test.loc[:, 'BsmtFinType1'].fillna("no")
test.loc[:, 'BsmtFinType2'] = test.loc[:, 'BsmtFinType2'].fillna("no")
test.loc[:, 'FireplaceQu' ] = test.loc[:, 'FireplaceQu' ].fillna("no")
test.loc[:, 'PoolQC'      ] = test.loc[:, 'PoolQC'      ].fillna("no")
test.loc[:, 'Fence'       ] = test.loc[:, 'Fence'       ].fillna("no")
test.loc[:, 'MiscFeature' ] = test.loc[:, 'MiscFeature' ].fillna("none")
test.shape
missing_columns = test.columns[test.isnull().any()]
print(missing_columns)
for column in missing_columns:
    if column != 'MSZoning':
        if test[column].dtype.name == 'object':
            test.loc[:, column] = test.loc[:, column].fillna("none")
        else:
            test.loc[:, column] = test.loc[:, column].fillna(0)
            
test.loc[:, 'MSZoning'] = test.loc[:, 'MSZoning'].fillna("none")
test.shape
test = test.replace({'Street'       : { 'Grvl' : 1, 'Pave' : 2 },
                       'Alley'        : { 'none' : 0, 'Grvl' : 1, 'Pave' : 2},
                       'ExterQual'    : { 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5 },
                       'ExterCond'    : { 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5 },
                       'BsmtQual'     : { 'no' : 0, 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5 },
                       'BsmtCond'     : { 'no' : 0, 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5 },
                       'BsmtExposure' : { 'no' : 0, 'No' : 1, 'Mn' : 2, 'Av' : 3, 'Gd' : 4 },
                       'BsmtFinType1' : { 'no' : 0, 'Unf' : 1, 'LwQ' : 2, 'Rec' : 3, 'BLQ' : 4, 'ALQ' : 5, 'GLQ' : 6 },
                       'BsmtFinType2' : { 'no' : 0, 'Unf' : 1, 'LwQ' : 2, 'Rec' : 3, 'BLQ' : 4, 'ALQ' : 5, 'GLQ' : 6 },
                       'HeatingQC'    : { 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5 },
                       'CentralAir'   : { 'N' : 0, 'Y' : 1 },
                       'KitchenQual'  : { 'none' : 0, 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5 },
                       'Functional'   : { 'none' : 0, 'Sal' : 1, 'Sev' : 2, 'Maj2' : 3, 'Maj1' : 4, 'Mod' : 5, 'Min2' : 6, 'Min1' : 7, 'Typ' : 8 },
                       'FireplaceQu'  : { 'no' : 0, 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5 },
                       'GarageQual'   : { 'none' : 0, 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5 },
                       'GarageCond'   : { 'none' : 0, 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5 },
                       'PavedDrive'   : { 'N' : 1, 'P': 2, 'Y' : 3},
                       'PoolQC'       : { 'no' : 0, 'Fa' : 1, 'TA' : 2, 'Gd' : 3, 'Ex' : 4 },
                       'Fence'        : { 'no' : 0, 'MnWw' : 1, 'GdWo' : 2, 'MnPrv' : 3, 'GdPrv' : 4 }
                      })

test.shape
test['SimpleOverallQual']   = test.OverallQual .replace({ 1 : 1, 2 : 1, 3: 1, 4 : 2, 5 : 2, 6 : 2, 7 : 3, 
                                                           8 : 3, 9 : 3, 10 : 3 })
test['SimpleOverallCond']   = test.OverallCond .replace({ 1 : 1, 2 : 1, 3: 1, 4 : 2, 5 : 2, 6 : 2, 7 : 3, 
                                                           8 : 3, 9 : 3, 10 : 3 })
test['SimpleExterQual' ]   = test.ExterQual   .replace({ 1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2 })
test['SimpleExterCond' ]   = test.ExterCond   .replace({ 1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2 })
test['SimpleBsmtQual'  ]   = test.BsmtQual    .replace({ 0 : 0, 1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2 })
test['SimpleBsmtCond'  ]   = test.BsmtCond    .replace({ 0 : 0, 1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2 })
test['SimpleBsmtExposure'] = test.BsmtExposure.replace({ 0 : 0, 1 : 1, 2 : 1, 3 : 2, 4 : 2 })
test['SimpleBsmtFinType1'] = test.BsmtFinType1.replace({ 0 : 0, 1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2, 6 : 2 })
test['SimpleBsmtFinType2'] = test.BsmtFinType2.replace({ 0 : 0, 1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2, 6 : 2 })
test['SimpleHeatingQC'   ] = test.HeatingQC   .replace({ 1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2 })
test['SimpleKitchenQual' ] = test.KitchenQual .replace({ 1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2 })
test['SimpleFunctional'  ] = test.Functional  .replace({ 0 : 0, 1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2 })
test['SimpleFireplaceQu' ] = test.FireplaceQu .replace({ 0 : 0, 1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2 })
test['SimpleGarageQual'  ] = test.GarageQual  .replace({ 0 : 0, 1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2 })
test['SimpleGarageCond'  ] = test.GarageCond  .replace({ 1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2 })
test['SimplePoolQC'      ] = test.PoolQC      .replace({ 0 : 0, 1 : 1, 2 : 1, 3 : 2, 4 : 2 })
test['SimpleFence'       ] = test.Fence       .replace({ 0 : 0, 1 : 1, 2 : 1, 3 : 2, 4 : 2 })

test.shape
test['SimpleGarageCond'].value_counts()
test['OverallScore']   = test['SimpleOverallQual'] * test['SimpleOverallCond']
test['ExterScore'  ]   = test['SimpleExterCond'] * test['SimpleExterQual']
test['BsmtScore'   ]   = test['SimpleBsmtCond'] * test['SimpleBsmtExposure'] * test['SimpleBsmtFinType1'] * test['SimpleBsmtFinType2']
test['GarageScore' ]   = test['SimpleGarageCond'] * test['SimpleGarageQual']
test['KitchenScore']   = test['SimpleKitchenQual'] * test['KitchenAbvGr']
test['FireplaceScore'] = test['FireplaceQu'] * test['Fireplaces']
test['PoolScore'] = test['PoolArea'] * test['SimplePoolQC']

test['OverallHouseSF'] = test['GrLivArea'] + test['TotalBsmtSF'] 
test['PorchSF'] = test['OpenPorchSF'] + test['EnclosedPorch'] + test['3SsnPorch'] \
+ test['ScreenPorch']
test["TotalBath"] = test["BsmtFullBath"] + (0.5 * test["BsmtHalfBath"]) + \
test["FullBath"] + (0.5 * test["HalfBath"])



test["HasMasVnr"] = test.MasVnrType.replace({"BrkCmn" : 1, "BrkFace" : 1, "CBlock" : 1, 
                                               "Stone" : 1, "None" : 0})

test["BoughtOffPlan"] = test.SaleCondition.replace({"Abnorml" : 0, "Alloca" : 0, "AdjLand" : 0, 
                                                      "Family" : 0, "Normal" : 0, "Partial" : 1})

test.shape
test['SquareOverallHouseSF'   ] = test['OverallHouseSF'   ]**2 
test['SquareOverallQual'      ] = test['OverallQual'      ]**2 
test['SquareGrLivArea'        ] = test['GrLivArea'        ]**2 
test['SquareSimpleOverallQual'] = test['SimpleOverallQual']**2 
test['SquareExterQual'        ] = test['ExterQual'        ]**2 
test['SquareTotalBath'        ] = test['TotalBath'        ]**2 
test['SquareKitchenQual'      ] = test['KitchenQual'      ]**2 
test['SquareGarageCars'       ] = test['GarageCars'       ]**2 
test['SquareSimpleExterQual'  ] = test['SimpleExterQual'  ]**2 

test['CubedOverallHouseSF'   ] = test['OverallHouseSF'    ]**3
test['CubedOverallQual'      ] = test['OverallQual'       ]**3
test['CubedGrLivArea'        ] = test['GrLivArea'         ]**3
test['CubedSimpleOverallQual'] = test['SimpleOverallQual' ]**3
test['CubedExterQual'        ] = test['ExterQual'         ]**3
test['CubedTotalBath'        ] = test['TotalBath'         ]**3
test['CubedKitchenQual'      ] = test['KitchenQual'       ]**3
test['CubedGarageCars'       ] = test['GarageCars'        ]**3
test['CubedSimpleExterQual'  ] = test['SimpleExterQual'   ]**3

test.shape
test['BsmtExposure'      ] = test['BsmtExposure'      ].apply(lambda x : np.log(x) if x != 0 else 0)
test['BsmtFinSF1'        ] = test['BsmtFinSF1'        ].apply(lambda x : np.log(x) if x != 0 else 0)
test['BsmtFinType2'      ] = test['BsmtFinType2'      ].apply(lambda x : np.log(x) if x != 0 else 0)
test['BsmtFinSF2'        ] = test['BsmtFinSF2'        ].apply(lambda x : np.log(x) if x != 0 else 0)
test['BsmtUnfSF'         ] = test['BsmtUnfSF'         ].apply(lambda x : np.log(x) if x != 0 else 0)
test['TotalBsmtSF'       ] = test['TotalBsmtSF'       ].apply(lambda x : np.log(x) if x != 0 else 0)
test['SimpleBsmtCond'    ] = test['SimpleBsmtCond'    ].apply(lambda x : np.log(x) if x != 0 else 0)
test['SimpleBsmtExposure'] = test['SimpleBsmtExposure'].apply(lambda x : np.log(x) if x != 0 else 0)
test['SimpleBsmtFinType2'] = test['SimpleBsmtFinType2'].apply(lambda x : np.log(x) if x != 0 else 0)
test['BsmtScore'         ] = test['BsmtScore'         ].apply(lambda x : np.log(x) if x != 0 else 0)
test['2ndFlrSF'          ] = test['2ndFlrSF'          ].apply(lambda x : np.log(x) if x != 0 else 0)
test['HalfBath'          ] = test['HalfBath'          ].apply(lambda x : np.log(x) if x != 0 else 0)
test['Alley'             ] = test['Alley'             ].apply(lambda x : np.log(x) if x != 0 else 0)
test['MasVnrArea'        ] = test['MasVnrArea'        ].apply(lambda x : np.log(x) if x != 0 else 0)
test['LowQualFinSF'      ] = test['LowQualFinSF'      ].apply(lambda x : np.log(x) if x != 0 else 0)
test['Fireplaces'        ] = test['Fireplaces'        ].apply(lambda x : np.log(x) if x != 0 else 0)
test['FireplaceScore'    ] = test['FireplaceScore'    ].apply(lambda x : np.log(x) if x != 0 else 0)
test['PoolArea'          ] = test['PoolArea'          ].apply(lambda x : np.log(x) if x != 0 else 0)
test['PoolQC'            ] = test['PoolQC'            ].apply(lambda x : np.log(x) if x != 0 else 0)
test['SimplePoolQC'      ] = test['SimplePoolQC'      ].apply(lambda x : np.log(x) if x != 0 else 0)
test['PoolScore'         ] = test['PoolScore'         ].apply(lambda x : np.log(x) if x != 0 else 0)
test['Fence'             ] = test['Fence'             ].apply(lambda x : np.log(x) if x != 0 else 0)
test['SimpleFence'       ] = test['SimpleFence'       ].apply(lambda x : np.log(x) if x != 0 else 0)
test['MiscVal'           ] = test['MiscVal'           ].apply(lambda x : np.log(x) if x != 0 else 0)
test['PorchSF'           ] = test['PorchSF'           ].apply(lambda x : np.log(x) if x != 0 else 0)

test.shape
categorical_features = test.select_dtypes(include = ["object"]).columns
print(categorical_features)
numerical_features = test.select_dtypes(exclude = ["object"]).columns
test_num = test[numerical_features]
test_cat = test[categorical_features]

test_cat = pd.get_dummies(test_cat)
test = pd.concat([test_num, test_cat], axis = 1)
for diff in diff_names:
    if diff not in test.columns:
        test[diff] = pd.Series(len(test['Fireplaces']), index=test.index)
        test[diff] = 0
test.shape
test_names = test.columns.get_values().tolist()
train_names = train.columns.get_values().tolist()

diff_set = list(set(test_names) - set(train_names))

test = test.drop(diff_set, axis=1)
test.shape
stdSc = StandardScaler()
test.loc[:, numerical_features] = stdSc.fit_transform(test.loc[:, numerical_features])
test_pred = lasso.predict(test)
test_pred = np.expm1(test_pred)
print(test_pred)
test = pd.read_csv('../input/test.csv')
col = ['SalePrice']
submission = pd.DataFrame(index = test['Id'], columns=col)
submission['SalePrice'] = test_pred
submission.shape
submission.to_csv('first_submission_housing.csv')
