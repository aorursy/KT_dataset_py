import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from sklearn import metrics

from scipy.special import boxcox1p

from sklearn import metrics

from scipy.stats import norm

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
train_df = pd.read_csv('../input/train.csv', index_col = [0])

test_df = pd.read_csv('../input/test.csv', index_col = [0])
'''

Here, 4 outliers are clearly seen. Two of these are partial sales hence

the large area and low sale price and the other two have large areas but

with sale prices well above the rest of the observations.



Note: Can see some evidence of heteroscedasticity here.

'''

sns.regplot(train_df['GrLivArea'], train_df['SalePrice'])

plt.title('Outliers in GrLivArea', fontsize = 14)

plt.xlabel('General Living Area', fontsize = 14)

plt.ylabel('Sale Price', fontsize = 14)

plt.show()
'''

Following the dataset's author's recommendations to remove any observation

with a living area of more than 4000 square feet, we only come across the

4 outliers mentioned above and not 5 as is mentioned in the documentation.

'''

train_df[train_df['GrLivArea'] > 4000].index
# These 4 outliers are removed.



train = train_df[train_df['GrLivArea'] <= 4000]
'''

From the plot below the regression plot for the axes has lower variability.

'''

sns.regplot(train['GrLivArea'], train['SalePrice'])

plt.title('Outliers remmoved from GrLivArea', fontsize = 14)

plt.xlabel('General Living Area', fontsize = 14)

plt.ylabel('Sale Price', fontsize = 14)

plt.show()
train = train[train['LotArea'] < 100000]

train = train[train['TotalBsmtSF'] < 3000]

train = train[train['1stFlrSF'] < 2500]

train = train[train['BsmtFinSF1'] < 2000]
X = pd.concat((train.iloc[:,:-1], test_df), sort = False)

y = train.iloc[:, -1]
numerical_df = X.select_dtypes(include = np.number)

categorical_df = X.select_dtypes(exclude = np.number)
'''

The plot below shows a skewed distribution for SalePrice or heteroscedasticity.



A normal distribution should follow the red line in the probability plot

on the right.

'''

plt.figure(figsize = (12,6))



plt.subplot(1,2,1)

sns.distplot(y, fit = norm)

plt.title('Distribution plot for SalePrice', fontsize = 14)

plt.xlabel('SalePrice', fontsize = 14)

plt.xticks(fontsize=10)

plt.yticks(fontsize=10)



plt.subplot(1,2,2)

stats.probplot(y, plot=plt)

plt.title('Probability plot for SalePrice', fontsize = 14)

plt.xlabel('Theoretical Quantiles', fontsize = 14)



plt.show()
'''

Since the above distribution look like a log-normal distribution,

a log transformation is done to SalePrice.



Below the plot shows a normal-like distribution after performing

a log + 1 transformation.



The probability plot of right follows the red line more closely.



The + 1 is to avoid taking a log of 0, which is undefined as well as ensuring the result is not negative. 

'''

plt.figure(figsize = (12,6))



plt.subplot(1,2,1)

sns.distplot(np.log1p(y), fit = norm)

plt.title('Distribution plot for log of SalePrice', fontsize = 14)

plt.xlabel('SalePrice', fontsize = 14)

plt.xticks(fontsize=10)

plt.yticks(fontsize=10)



plt.subplot(1,2,2)

stats.probplot(np.log1p(y), plot=plt)

plt.title('Probability plot for log of SalePrice', fontsize = 14)

plt.xlabel('Theoretical Quantiles', fontsize = 14)



plt.show()
'''

Performing a log transformation on the target variable.

'''

y = np.log1p(y)
plt.figure(figsize = (14,6))



plt.subplot(1,2,1)

sns.regplot(train['GrLivArea'], train['SalePrice'])

plt.title('Before Log Transformation', fontsize = 14)

plt.xlabel('Above Ground Living Area', fontsize = 14)

plt.ylabel('Sale Price', fontsize = 14)



plt.subplot(1,2,2)

sns.regplot(train['GrLivArea'], y)

plt.title('After Log Transformation', fontsize = 14)

plt.xlabel('Above Ground Living Area', fontsize = 14)

plt.ylabel('Sale Price', fontsize = 14)

plt.show()
plt.figure(figsize=(15, 15))

sns.heatmap(numerical_df.corr(), cmap = 'coolwarm', square = True, vmax=.8)

plt.xticks(fontsize=11)

plt.yticks(fontsize=11)

plt.show()
'''

Below the feature with a correlation of 0.8 or higher are identified.



Filtered for correlation less than 1 to rule out self-correlated instances.

'''

unstacked_corr_df = numerical_df.corr().abs().unstack()

sorted_corr_df = unstacked_corr_df.sort_values()

sorted_corr_df[(sorted_corr_df > 0.8) & (sorted_corr_df < 1)]
numerical_df[['GarageArea', 'GarageCars']].corrwith(y)
temp_df = numerical_df[['YearBuilt', 'GarageYrBlt']].dropna()

equal_cols = len(temp_df[temp_df['YearBuilt'] == temp_df['GarageYrBlt']].index)

equal_cols/len(temp_df.index)*100
print('Number of missing values for GarageYrBlt:', numerical_df['GarageYrBlt'].isnull().sum())

print('Number of missing values for YearBuilt:', numerical_df['YearBuilt'].isnull().sum())
numerical_df[['GrLivArea', 'TotRmsAbvGrd']].corrwith(y)
'''

Creating a list of columns to drop.

'''

drop_cols = ['GarageArea', 'GarageYrBlt']
X[X.apply(lambda x: 79 - x.count(), axis=1) > 20].apply(lambda x: 79 - x.count(), axis=1)
null_cols = X.columns[X.isnull().any()]

missing_data = (X[null_cols].isnull().sum()/len(X)*100).sort_values(ascending = False)



plt.figure(figsize = (12,6))

plt.bar(missing_data.index, missing_data)

plt.title('Percentage of missing values in each feature', fontsize = 14)

plt.ylabel('Percentage', fontsize = 14)

plt.xticks(rotation = 90)

plt.show()
X.drop(['PoolQC', 'MiscFeature', 'Fence'], axis = 1, inplace = True)
cols = X.columns[X.isnull().any()]



print('Number of numeric columns with missing data', len(X[cols].select_dtypes(include = np.number).columns))

print('Number of categorical columns with missing data', len(X[cols].select_dtypes(exclude = np.number).columns))
X_float_missing = X[cols].select_dtypes(include=np.number)

X_float_missing.isnull().sum()
plt.figure(figsize=(15,8))

sns.boxplot(X['Neighborhood'], X['LotFrontage'], width=0.7, linewidth=0.8)

plt.xticks(rotation = 60)

plt.title('Boxplot of LotFrontage per Neighborhood', fontsize = 14)

plt.xlabel('Neighborhood', fontsize = 14)

plt.ylabel('LotFrontage', fontsize = 14)

plt.show()
X['LotFrontage'] = X.groupby('Neighborhood')['LotFrontage'].apply(lambda x: x.fillna(x.mean()))
X[['MasVnrType', 'MasVnrArea']].isnull().sum()
X[(X['MasVnrType'].isnull()) & (X['MasVnrArea'].notnull())][['MasVnrType', 'MasVnrArea']]
X['MasVnrType'].value_counts(dropna = False)
X.at[2611, 'MasVnrType'] = 'BrkFace'
X['MasVnrType'] = X['MasVnrType'].fillna('NoVnr')

X['MasVnrArea'] = X['MasVnrArea'].fillna(0)
X[['BsmtCond', 'BsmtQual', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'TotalBsmtSF']].isnull().sum()
X[(X['BsmtExposure'].isnull()) & (X['BsmtFinType1'].notnull())][['BsmtExposure', 'BsmtFinType1']]
X['BsmtExposure'].mode()[0]
X.update(X[(X['BsmtExposure'].isnull()) & (X['BsmtFinType1'].notnull())]['BsmtExposure'].fillna(X['BsmtExposure'].mode()[0]))
X[(X['BsmtCond'].isnull()) & (X['BsmtQual'].notnull())][['BsmtQual','BsmtCond']]
X.update(X[(X['BsmtCond'].isnull()) & (X['BsmtQual'] == 'Gd')]['BsmtCond'].fillna('Gd'))
X.update(X[(X['BsmtCond'].isnull()) & (X['BsmtQual'] == 'TA')]['BsmtCond'].fillna('TA'))
X[(X['BsmtCond'].notnull()) & (X['BsmtQual'].isnull())][['BsmtQual','BsmtCond']]
X.update(X[(X['BsmtQual'].isnull()) & (X['BsmtCond'] == 'Fa')]['BsmtCond'].fillna('Fa'))
X.update(X[(X['BsmtQual'].isnull()) & (X['BsmtCond'] == 'TA')]['BsmtCond'].fillna('TA'))
X.update(X[['BsmtCond', 'BsmtFinType1', 'BsmtFinType2', 'BsmtExposure']].fillna('NoBsmt'))
X.update(X[['TotalBsmtSF', 'BsmtQual']].fillna(0))
cols = X.columns[X.isnull().any()]

X_float_missing = X[cols].select_dtypes(include=['float64'])

X_float_missing.isnull().sum()
X[X['BsmtFullBath'].isnull()][['BsmtFullBath', 'BsmtHalfBath', 'BsmtQual','TotalBsmtSF']]
X[['BsmtFullBath', 'BsmtHalfBath']] = X[['BsmtFullBath', 'BsmtHalfBath']].fillna(0)
cols = X.columns[X.isnull().any()]

X[cols].isnull().sum()
X[X['BsmtFinSF1'].isnull()][['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']]
X[['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF']] = X[['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF']].fillna(0)
X.update(X[['Alley','FireplaceQu','GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']].fillna('None'))
cols = X.columns[X.isnull().any()]

X[cols].isnull().sum()
X[X['GarageCars'].isnull()][['GarageCars', 'GarageArea']]
X['GarageCars'] = X['GarageCars'].fillna(X['GarageCars'].mode()[0])
plt.subplots(figsize =(12, 6))



plt.subplot(1, 2, 1)

sns.countplot('Utilities', data = X).set_title('Train data - Utilities')



plt.subplot(1, 2, 2)

sns.countplot('Utilities', data = test_df).set_title('Test data - Utilities')



plt.show()
'''

Adding column to drop to drop list.

'''

drop_cols = drop_cols + ['Utilities']
cols = ['MSZoning', 'Functional', 'SaleType', 'KitchenQual', 'Electrical', 'Exterior1st', 'Exterior2nd']

for col in cols:

    X.update(X[col].fillna(X[col].mode()[0]))
X.drop(drop_cols, axis = 1, inplace = True)
cols = X.columns[X.isnull().any()]

X[cols].isnull().sum()
X.drop(['MiscVal'], axis = 1, inplace = True)
len(X[X['PoolArea'] == 0].index)/len(X.index)*100
X.drop(['PoolArea'], axis = 1 , inplace = True)
fig, axs = plt.subplots(ncols=2, nrows=16, figsize=(12, 80))



for i, feature in enumerate(X[:train.shape[0]].select_dtypes(include=np.number).columns, 1):    

    plt.subplot(16, 2, i)

    plt.scatter(x=feature, y= y, data=(X[:train.shape[0]].select_dtypes(include=np.number)))

        

    plt.xlabel('{}'.format(feature), size=15)

    plt.ylabel('SalePrice', size=15, labelpad=12.5)

    

    for j in range(2):

        plt.tick_params(axis='x', labelsize=12)

        plt.tick_params(axis='y', labelsize=12)

    

plt.show()
sns.pairplot(X[['GrLivArea', 'LotArea', 'TotalBsmtSF']])

plt.show()
cols = ['LotArea', 'GrLivArea', 'TotalBsmtSF']



# Using boxcox + 1 to transform the features.

for col in cols:

    X[col] = boxcox1p(X[col], 0.1)
sns.pairplot(X[['GrLivArea', 'LotArea', 'TotalBsmtSF']])

plt.show()
X['TotalBath'] = X['FullBath'] + X['HalfBath']*0.5 + X['BsmtFullBath'] + X['BsmtHalfBath']*0.5

X['TotalFlrSF'] = X['1stFlrSF'] + X['2ndFlrSF']

X['BsmtFinSF'] = X['BsmtFinSF1'] + X['BsmtFinSF2']
# Old features are dropped



comb_cols_drop = ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath', '1stFlrSF', '2ndFlrSF', 'BsmtFinSF1', 'BsmtFinSF2']

for col in comb_cols_drop:

    X.drop([col], axis =1, inplace = True)
X['LandContour'] = X['LandContour'].replace(dict(Lvl=4, Bnk=3, HLS=2, Low=1))

X['LandSlope'] = X['LandSlope'].replace(dict(Gtl=3, Mod=2, Sev=1))

X['ExterQual'] = X['ExterQual'].replace(dict(Ex=5, Gd=4, TA=3, Fa=2, Po=1))

X['ExterCond'] = X['ExterCond'].replace(dict(Ex=5, Gd=4, TA=3, Fa=2, Po=1))



X['BsmtQual'] = X['BsmtQual'].replace(dict(Ex=5, Gd=4, TA=3, Fa=2, Po=1, NoBsmt=0))

X['BsmtCond'] = X['BsmtCond'].replace(dict(Ex=5, Gd=4, TA=3, Fa=2, Po=1, NoBsmt=0))

X['BsmtExposure'] = X['BsmtExposure'].replace(dict(Gd=4, Av=3, Mn=2, No=1, NoBsmt=0))

X['BsmtFinType1'] = X['BsmtFinType1'].replace(dict(GLQ=6, ALQ=5, BLQ=4, Rec=3, LwQ=2, Unf=1, NoBsmt=0))

X['BsmtFinType2'] = X['BsmtFinType2'].replace(dict(GLQ=6, ALQ=5, BLQ=4, Rec=3, LwQ=2, Unf=1, NoBsmt=0))



X['HeatingQC'] = X['HeatingQC'].replace(dict(Ex=5, Gd=4, TA=3, Fa=2, Po=1))

X['CentralAir'] = X['CentralAir'].replace(dict(Y=1, N=0))

X['KitchenQual'] = X['KitchenQual'].replace(dict(Ex=5, Gd=4, TA=3, Fa=2, Po=1))

X['Functional'] = X['Functional'].replace(dict(Typ=8, Min1=7, Min2=6, Mod=5, Maj1=4, Maj2=3, Sev=2, Sal=1))

X['FireplaceQu'] = X['FireplaceQu'].replace(dict(Ex=5, Gd=4, TA=3, Fa=2, Po=1))

X['FireplaceQu'] = X['FireplaceQu'].replace('None', 0)



X['GarageQual'] = X['GarageQual'].replace(dict(Ex=5, Gd=4, TA=3, Fa=2, Po=1))

X['GarageQual'] = X['GarageQual'].replace('None', 0)

X['GarageCond'] = X['GarageCond'].replace(dict(Ex=5, Gd=4, TA=3, Fa=2, Po=1))

X['GarageCond'] = X['GarageCond'].replace('None', 0)

X['GarageFinish'] = X['GarageFinish'].replace(dict(Fin=3, RFn=2, Unf=1))

X['GarageFinish'] = X['GarageFinish'].replace('None', 0)





X['LotShape'] = X['LotShape'].replace(dict(Reg=4, IR1=3, IR2=2, IR3=1))

X['PavedDrive'] = X['PavedDrive'].replace(dict(Y=3, P=2, N=1))
X['MSSubClass'] = X['MSSubClass'].astype('category')

X['MoSold'] = X['MoSold'].astype('category')

X['YrSold'] = X['YrSold'].astype('category')
X = pd.get_dummies(X, drop_first=True)
X_tr = X[:train.shape[0]] # Modelling set

X_t = X[train.shape[0]:] # Prediction set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_tr, y, test_size=0.20, shuffle=False)
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Ridge



ridge = Ridge()

parameters = {'alpha': [1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}

ridge_regressor = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=10, iid = True)



ridge_regressor.fit(X_train, y_train)
# RMSLE for train set and test set

print('RMSLE tests')



train_rr = ridge_regressor.predict(X_train)

print('Ridge Train:', np.sqrt(metrics.mean_squared_log_error(np.expm1(y_train), np.expm1(train_rr))))



test_rr = ridge_regressor.predict(X_test)

print('Ridge Test:', np.sqrt(metrics.mean_squared_log_error(np.expm1(y_test), np.expm1(test_rr))))
yhat = ridge_regressor.predict(X_train)

SS_Residual = sum((y_train-yhat)**2)

SS_Total = sum((y-np.mean(y))**2)

r_squared = 1 - (float(SS_Residual))/SS_Total

adjusted_r_squared = 1 - (1-r_squared)*(len(y)-1)/(len(y)-X.shape[1]-1)

print('Train Scores')

print(r_squared, adjusted_r_squared)
yhat = ridge_regressor.predict(X_test)

SS_Residual = sum((y_test-yhat)**2)

SS_Total = sum((y-np.mean(y))**2)

r_squared = 1 - (float(SS_Residual))/SS_Total

adjusted_r_squared = 1 - (1-r_squared)*(len(y)-1)/(len(y)-X.shape[1]-1)

print('Test Scores')

print(r_squared, adjusted_r_squared)
from sklearn import linear_model



parameters = {'alpha': np.arange(0.0001,0.005, 0.0001)}

ls = linear_model.Lasso()

lasso = GridSearchCV(ls, parameters, scoring='neg_mean_squared_error', cv=10, iid = True)



lasso.fit(X_train, y_train)
# RMSLE for train set and test set

print('RMSLE tests')



train_ls = lasso.predict(X_train)

print('Lasso Train:', np.sqrt(metrics.mean_squared_log_error(np.expm1(y_train), np.expm1(train_ls))))



test_ls = lasso.predict(X_test)

print('Lasso Test:', np.sqrt(metrics.mean_squared_log_error(np.expm1(y_test), np.expm1(test_ls))))
from sklearn.ensemble import RandomForestRegressor



param_grid = {'bootstrap': [True, False], 'max_depth': [5, 10, 15, 20, 30, 40, 60, 80, 100, 120],

    'max_features': [2, 3, 5, 7, 10, 13, 16, 20],

    'min_samples_leaf': [3, 4, 5, 6, 7],

    'min_samples_split': [8, 10, 12, 14, 16],

    'n_estimators': [100, 200, 300, 600, 1000]}



rndm_frst = RandomForestRegressor()  

grid_search = GridSearchCV(estimator = rndm_frst, param_grid = param_grid, 

                          cv = 5, verbose = 2, scoring='neg_mean_squared_error')



rndm_frst.fit(X_train, y_train)
# RMSLE for train set and test set

print('RMSLE tests')



train_rf = rndm_frst.predict(X_train)

print('RF Train:', np.sqrt(metrics.mean_squared_log_error(np.expm1(y_train), np.expm1(train_rf))))



test_rf = rndm_frst.predict(X_test)

print('RF Test:', np.sqrt(metrics.mean_squared_log_error(np.expm1(y_test), np.expm1(test_rf))))
prediction = np.expm1(ridge_regressor.predict(X_t))
sub_df = pd.DataFrame({"id":test_df.index, "SalePrice":prediction})

sub_df.to_csv("ridge_reg.csv", index = False)