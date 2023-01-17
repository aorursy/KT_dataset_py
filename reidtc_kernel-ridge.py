import os

import pylab

import numpy as np

import pandas as pd

from math import sqrt

import seaborn as sns

from scipy import stats

import statsmodels.api as sm

from scipy.stats import zscore

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.linear_model import Ridge

from sklearn.model_selection import KFold

from sklearn.kernel_ridge import KernelRidge

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer

from sklearn.preprocessing import MinMaxScaler, RobustScaler

from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

submission = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
print(train.shape)

print(test.shape)
train.head()
test.head()
train['MSSubClass'] = train['MSSubClass'].astype(str)

test['MSSubClass'] = test['MSSubClass'].astype(str)



train['MoSold'] = train['MoSold'].astype(str)

test['MoSold'] = test['MoSold'].astype(str)



train['YrSold'] = train['YrSold'].astype(str)

test['YrSold'] = test['YrSold'].astype(str)



train['GarageYrBlt'] = train['GarageYrBlt'].astype(str)

test['GarageYrBlt'] = test['GarageYrBlt'].astype(str)



train['GarageCars'] = train['GarageCars'].astype(str)

test['GarageCars'] = test['GarageCars'].astype(str)



train['OverallQual'] = train['OverallQual'].astype(str)

test['OverallQual'] = test['OverallQual'].astype(str)



train['OverallCond'] = train['OverallCond'].astype(str)

test['OverallCond'] = test['OverallCond'].astype(str)



train['BsmtFullBath'] = train['BsmtFullBath'].astype(str)

test['BsmtFullBath'] = test['BsmtFullBath'].astype(str)



train['BsmtHalfBath'] = train['BsmtHalfBath'].astype(str)

test['BsmtHalfBath'] = test['BsmtHalfBath'].astype(str)



train['FullBath'] = train['FullBath'].astype(str)

test['FullBath'] = test['FullBath'].astype(str)



train['HalfBath'] = train['HalfBath'].astype(str)

test['HalfBath'] = test['HalfBath'].astype(str)



train['BedroomAbvGr'] = train['BedroomAbvGr'].astype(str)

test['BedroomAbvGr'] = test['BedroomAbvGr'].astype(str)



train['KitchenAbvGr'] = train['KitchenAbvGr'].astype(str)

test['KitchenAbvGr'] = test['KitchenAbvGr'].astype(str)



train['TotRmsAbvGrd'] = train['TotRmsAbvGrd'].astype(str)

test['TotRmsAbvGrd'] = test['TotRmsAbvGrd'].astype(str)



train['Fireplaces'] = train['Fireplaces'].astype(str)

test['Fireplaces'] = test['Fireplaces'].astype(str)
submission.head()
cols = train.columns

cols = cols.drop(['Id','SalePrice','PoolArea','LowQualFinSF','3SsnPorch',

                  'MiscVal','ScreenPorch','EnclosedPorch','BsmtFinSF2'])



train_y = pd.DataFrame(data=train['SalePrice'], columns=['SalePrice'])

train_x = pd.DataFrame(data=train[cols], columns=cols)
train_y.isnull().any()
test_x = pd.DataFrame(data=test[cols], columns=cols)
cols_numeric = []

cols_discrete = []

for x in train_x.columns:

    if not train_x[x].dtype == object:

        cols_numeric = np.append(cols_numeric, x)

    else:

        cols_discrete = np.append(cols_discrete, x)



print(cols_numeric, '\n')

print(cols_discrete)
# convert to train_x and add test_x

train_x['Alley'] = np.where(train_x['Alley'].isna(), 'None', train_x['Alley'])

train_x['MasVnrType'] = np.where(train_x['MasVnrType'].isna(), 'None', train_x['MasVnrType'])

train_x['BsmtQual'] = np.where(train_x['BsmtQual'].isna(), 'None', train_x['BsmtQual'])

train_x['BsmtCond'] = np.where(train_x['BsmtCond'].isna(), 'None', train_x['BsmtCond'])

train_x['BsmtExposure'] = np.where(train_x['BsmtExposure'].isna(), 'None', train_x['BsmtExposure'])

train_x['BsmtFinType1'] = np.where(train_x['BsmtFinType1'].isna(), 'None', train_x['BsmtFinType1'])

train_x['BsmtFinType2'] = np.where(train_x['BsmtFinType2'].isna(), 'None', train_x['BsmtFinType2'])

train_x['FireplaceQu'] = np.where(train_x['FireplaceQu'].isna(), 'None', train_x['FireplaceQu'])

train_x['GarageType'] = np.where(train_x['GarageType'].isna(), 'None', train_x['GarageType'])

train_x['GarageQual'] = np.where(train_x['GarageQual'].isna(), 'None', train_x['GarageQual'])

train_x['GarageCond'] = np.where(train_x['GarageCond'].isna(), 'None', train_x['GarageCond'])

train_x['GarageYrBlt'] = np.where(train_x['GarageYrBlt'].isna(), 'None', train_x['GarageYrBlt'])

train_x['GarageFinish'] = np.where(train_x['GarageFinish'].isna(), 'None', train_x['GarageFinish'])

train_x['PoolQC'] = np.where(train_x['PoolQC'].isna(), 'None', train_x['PoolQC'])

train_x['Fence'] = np.where(train_x['Fence'].isna(), 'None', train_x['Fence'])

train_x['MiscFeature'] = np.where(train_x['MiscFeature'].isna(), 'None', train_x['MiscFeature'])
test_x['Alley'] = np.where(test_x['Alley'].isna(), 'None', test_x['Alley'])

test_x['MasVnrType'] = np.where(test_x['MasVnrType'].isna(), 'None', test_x['MasVnrType'])

test_x['BsmtQual'] = np.where(test_x['BsmtQual'].isna(), 'None', test_x['BsmtQual'])

test_x['BsmtCond'] = np.where(test_x['BsmtCond'].isna(), 'None', test_x['BsmtCond'])

test_x['BsmtExposure'] = np.where(test_x['BsmtExposure'].isna(), 'None', test_x['BsmtExposure'])

test_x['BsmtFinType1'] = np.where(test_x['BsmtFinType1'].isna(), 'None', test_x['BsmtFinType1'])

test_x['BsmtFinType2'] = np.where(test_x['BsmtFinType2'].isna(), 'None', test_x['BsmtFinType2'])

test_x['FireplaceQu'] = np.where(test_x['FireplaceQu'].isna(), 'None', test_x['FireplaceQu'])

test_x['GarageType'] = np.where(test_x['GarageType'].isna(), 'None', test_x['GarageType'])

test_x['GarageQual'] = np.where(test_x['GarageQual'].isna(), 'None', test_x['GarageQual'])

test_x['GarageCond'] = np.where(test_x['GarageCond'].isna(), 'None', test_x['GarageCond'])

test_x['GarageYrBlt'] = np.where(test_x['GarageYrBlt'].isna(), 'None', test_x['GarageYrBlt'])

test_x['GarageFinish'] = np.where(test_x['GarageFinish'].isna(), 'None', test_x['GarageFinish'])

test_x['PoolQC'] = np.where(test_x['PoolQC'].isna(), 'None', test_x['PoolQC'])

test_x['Fence'] = np.where(test_x['Fence'].isna(), 'None', test_x['Fence'])

test_x['MiscFeature'] = np.where(test_x['MiscFeature'].isna(), 'None', test_x['MiscFeature'])
for col in cols_numeric:

    print(train_x[col].describe(), '\n')
for col in cols_discrete:

    print(train_x[col].describe(), '\n')
for col in cols_numeric:

    plt.plot(train_x[col], train_y['SalePrice'], 'o')

    plt.title(col)

    plt.show()
outliers = train_x.copy()

outliers['SalePrice'] = train_y.copy()

no_outliers = pd.DataFrame()



lotfront = outliers['LotFrontage'].mean()+(outliers['LotFrontage'].std()*3)

lotarea = outliers['LotArea'].mean()+(outliers['LotArea'].std()*3)

bfinsf1 = outliers['BsmtFinSF1'].mean()+(outliers['BsmtFinSF1'].std()*3)

tbsf = outliers['TotalBsmtSF'].mean()+(outliers['TotalBsmtSF'].std()*3)

fsf = outliers['1stFlrSF'].mean()+(outliers['1stFlrSF'].std()*3)

gla = outliers['GrLivArea'].mean()+(outliers['GrLivArea'].std()*3)



no_outliers = outliers.loc[(outliers['LotFrontage'] < lotfront) &

                       (outliers['LotArea'] < lotarea) &

                       (outliers['BsmtFinSF1'] < bfinsf1) &

                       (outliers['TotalBsmtSF'] < tbsf) &

                       (outliers['1stFlrSF'] < fsf) &

                       (outliers['GrLivArea'] < gla)]



del(train_y)

#reindex these...

train_y = pd.DataFrame(data = no_outliers['SalePrice'], columns=['SalePrice'])

train_x = no_outliers.drop(['SalePrice'], axis=1)
print(train_x.shape)

print(train_y.index)

print(outliers.index)
train_x.isnull().values.any()
test_x.isnull().values.any()
train_x['MasVnrArea'] = np.where(train_x['MasVnrType'] == 'None', 0, train_x['MasVnrArea'])

test_x['MasVnrArea'] = np.where(test_x['MasVnrType'] == 'None', 0, test_x['MasVnrArea'])



train_x['BsmtFinSF1'] = np.where(train_x['BsmtFinType1'] == 'None', 0, train_x['BsmtFinSF1'])

test_x['BsmtFinSF1'] = np.where(test_x['BsmtFinType1'] == 'None', 0, test_x['BsmtFinSF1'])



train_x['BsmtUnfSF'] = np.where((train_x['BsmtFinType1'] == 'None') & 

                               (train_x['BsmtFinType2'] == 'None'), 0, train_x['BsmtUnfSF'])

test_x['BsmtUnfSF'] = np.where((test_x['BsmtFinType1'] == 'None') & 

                               (test_x['BsmtFinType2'] == 'None'), 0, test_x['BsmtUnfSF'])



train_x['2ndFlrSF'] = np.where((train_x['HouseStyle'] == '1Story'), 0, train_x['2ndFlrSF'])

test_x['2ndFlrSF'] = np.where((test_x['HouseStyle'] == '1Story'), 0, test_x['2ndFlrSF'])



train_x['GarageCars'] = np.where(train_x['GarageType'] == 'None', 'None', train_x['GarageCars'])

test_x['GarageCars'] = np.where(test_x['GarageType'] == 'None', 'None', test_x['GarageCars'])



train_x['GarageArea'] = np.where(train_x['GarageType'] == 'None', 0, train_x['GarageArea'])

test_x['GarageArea'] = np.where(test_x['GarageType'] == 'None', 0, test_x['GarageArea'])
imp = IterativeImputer(max_iter=10, random_state=0)



train_x[cols_numeric] = imp.fit_transform(train_x[cols_numeric])

test_x[cols_numeric] = imp.fit_transform(test_x[cols_numeric])
print(train_x.isnull().any().unique())

print(test_x.isnull().any().unique())
na_drop = train_x

na_drop['SalePrice'] = train_y



na_drop = na_drop.dropna(axis=0)



train_y = pd.DataFrame(data=na_drop['SalePrice'], columns=['SalePrice'])

print(train_y)

train_x = na_drop.drop(['SalePrice'], axis=1)
print(train_x.isnull().any().unique())

print(train_y.isnull().any().unique())
test_x = test_x.fillna(test_x.mode())
train_y.describe()
# the histogram of the data

n, bins, patches = plt.hist(train_y['SalePrice'], 50, density=True, alpha=0.75)



plt.xlabel('Sale Price')

plt.ylabel('Probability')

plt.title('Histogram of Sale Price')



plt.show()
train_y['SalePrice'] = np.log(train_y['SalePrice']+1)

print(train_y.isnull().any())
# the histogram of the data

n, bins, patches = plt.hist(train_y['SalePrice'], 50, density=True, alpha=0.75)



plt.xlabel('Sale Price')

plt.ylabel('Probability')

plt.title('Histogram of Sale Price')



plt.show()
for col in cols_numeric:

    n, bins, patches = plt.hist(train_x[col], 50, density=True, alpha=0.75)

    plt.title(col)

    plt.show()
to_log = ['LotFrontage',

          'LotArea',

          'MasVnrArea',

          'BsmtFinSF1',

          'BsmtUnfSF',

          '1stFlrSF',

          'GrLivArea',

          'WoodDeckSF',

          'OpenPorchSF']



train_x[to_log] = np.log(train_x[to_log]+1)

test_x[to_log] = np.log(test_x[to_log]+1)
for col in to_log:

    n, bins, patches = plt.hist(train_x[col], 50, density=True, alpha=0.75)

    plt.title(col)

    plt.show()
train_x['hasmva'] = np.where(train_x['MasVnrArea'] == 0, False, True)

test_x['hasmva'] = np.where(test_x['MasVnrArea'] == 0, False, True)



train_x['hasbf1'] = np.where(train_x['BsmtFinSF1'] == 0, False, True)

test_x['hasbf1'] = np.where(test_x['BsmtFinSF1'] == 0, False, True)



train_x['hasbu'] = np.where(train_x['BsmtUnfSF'] == 0, False, True)

test_x['hasbu'] = np.where(test_x['BsmtUnfSF'] == 0, False, True)



train_x['hasdeck'] = np.where(train_x['WoodDeckSF'] == 0, False, True)

test_x['hasdeck'] = np.where(test_x['WoodDeckSF'] == 0, False, True)



train_x['hasporch'] = np.where(train_x['OpenPorchSF'] == 0, False, True)

test_x['hasporch'] = np.where(test_x['OpenPorchSF'] == 0, False, True)



train_x['hasbase'] = np.where(train_x['TotalBsmtSF'] == 0, False, True)

test_x['hasbase'] = np.where(test_x['TotalBsmtSF'] == 0, False, True)



train_x['has2nd'] = np.where(train_x['2ndFlrSF'] == 0, False, True)

test_x['has2nd'] = np.where(test_x['2ndFlrSF'] == 0, False, True)



train_x['hasgarage'] = np.where(train_x['GarageArea'] == 0, False, True)

test_x['hasgarage'] = np.where(test_x['GarageArea'] == 0, False, True)
print(train_x.shape)

print(test_x.shape)

split = train_x.shape[0]

print(split)

all_data = train_x.append(test_x, sort=False)

print(all_data.shape)
temp = pd.get_dummies(all_data[cols_discrete])

print(temp.shape)

all_data = all_data.drop(cols_discrete, axis=1)

print(all_data.shape)

all_data = pd.concat([all_data, temp], axis=1, sort=False)
print(all_data.shape)

train_x = all_data[:split]

test_x = all_data[split:]

print(train_x.shape)

print(test_x.shape)
for col in cols_numeric:

    train_x[col+'2'] = train_x[col]**2

    test_x[col+'2'] = test_x[col]**2
for col in cols_numeric:

    plt.plot(train_x[col], train_y['SalePrice'], 'o')

    plt.title(col)

    plt.show()
#to avoid data leakage

scaler = MinMaxScaler()

all_cols = train_x.columns

scaler.fit(train_x)

train_x = pd.DataFrame(data=scaler.transform(train_x), columns=all_cols)

test_x = pd.DataFrame(data=scaler.transform(test_x), columns=all_cols)
k = 5



X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.3, random_state=123)



kr_param_grid = {"alpha": [100, 50, 25, 10, 5, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],

                "kernel": ['linear','rbf','poly'],

                "gamma": [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]}

#

# # ridge works best

kr = GridSearchCV(KernelRidge(), cv=k, param_grid=kr_param_grid)



kr.fit(X_train, np.asarray(y_train).ravel())



train_rmse = sqrt(mean_squared_error(y_train, kr.predict(X_train)))

test_rmse = sqrt(mean_squared_error(y_test, kr.predict(X_test)))



print(train_rmse)

print(test_rmse)
print(kr.best_params_)
kf = KFold(n_splits=k, shuffle=False, random_state=42)

reg = KernelRidge(alpha=kr.best_params_['alpha'], kernel=kr.best_params_['kernel'], gamma=kr.best_params_['gamma'])

i=1

validation = pd.DataFrame()

results = pd.DataFrame()

resid = pd.DataFrame()



for train_index, test_index in kf.split(train_x):

#     print("TRAIN:", train_index, "TEST:", test_index)

    X_train, X_test = np.asarray(train_x)[train_index], np.asarray(train_x)[test_index]

    y_train, y_test = np.asarray(train_y)[train_index], np.asarray(train_y)[test_index]

    

    reg.fit(X_train, np.asarray(y_train).ravel())

#     print(reg.predict(test_x))

    temp_val = pd.DataFrame(data=[[sqrt(mean_squared_error(y_train, reg.predict(X_train))),

                                  sqrt(mean_squared_error(y_test, reg.predict(X_test)))]], 

                            columns=['train','test'])

    validation = pd.concat([validation,temp_val], axis=0)

    

    temp_resid = pd.DataFrame(data=reg.predict(train_x), columns=['res{0}'.format(i)])

    resid = pd.concat([resid, temp_resid], axis=1)

    

    temp_submission = pd.DataFrame(data=reg.predict(test_x), columns=['res{0}'.format(i)])

    results = pd.concat([results, temp_submission], axis=1)

    i+=1
validation.index = range(1,k+1)
plt.plot(validation['train'])

plt.plot(validation['test'])

plt.title('RMSE')

plt.show()
print(validation['train'].mean())

print(validation['test'].mean())
submission['SalePrice'] = np.exp(results.mean(axis=1))-1
train_y = train_y.reset_index(drop=True)
train_y.head()
df_res = pd.DataFrame(data=resid.mean(axis=1), columns=['pred'])

print(train_x.index)

print(train_y.shape)



residuals = pd.DataFrame(data=zscore(train_y['SalePrice'][train_y['SalePrice'].notnull()] - df_res['pred'][df_res['pred'].notnull()]))

residuals = residuals.reset_index(drop=True)
residuals.head()
print(np.mean(residuals))

print(np.std(residuals))
plt.plot(df_res['pred'], residuals, 'o')
print(df_res.index)

print(train_y.index)

print(residuals.index.astype(int))
plt.plot(df_res['pred'], train_y, 'o')
stats.probplot(residuals[0], dist="norm", plot=plt)

plt.show()
submission.head()
submission.to_csv('submission.csv', encoding='utf-8', index=False)