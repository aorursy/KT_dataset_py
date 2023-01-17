## import libraries for EDA

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
## load data, for EDA we'll only use train data

train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

pd.set_option('display.max_columns', 100) # Show all columns when looking at dataframe

pd.options.display.float_format='{:,.3f}'.format
test.shape
train.info()
train.describe()
## divide data in categorical and continuos

## I put sale price in both to see how our target relates to categorical and continuos variables 

df_num = train[['LotFrontage','LotArea','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1',

                'BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea',

                'TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','GarageArea','WoodDeckSF',

                'OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal',

                'YrSold','SalePrice']].copy(deep=True)

df_cat = train[['MSSubClass','MSZoning','Street','Alley','LotShape','LandContour','Utilities',

                'LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle',

                'OverallQual','OverallCond','RoofStyle','RoofMatl','Exterior1st','Exterior2nd',

                'MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure',

                'BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical',

                'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',

                'KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual',

                'GarageCond','PavedDrive','PoolQC','Fence','MiscFeature','MoSold','SaleType',

                'SaleCondition','SalePrice']].copy(deep=True)
corr = df_num.corr()

corr
sns.heatmap(corr, cmap='YlGnBu');
## set the style for plots

style = {'axes.facecolor': 'white',

         'ytick.left': False,

         'axes.spines.left': True,

         'axes.spines.right': False,

         'axes.spines.top': False}

palette = 'Set2'

sns.set(style=style, palette=palette)
vars_ = ['GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','TotRmsAbvGrd','YearBuilt','YearRemodAdd']

for i in vars_:

    ax = sns.distplot(df_num[i])

    ax.set(title=i, xlabel='', yticklabels=[])

    plt.show()

    ax = sns.regplot(x=df_num[i], y=df_num.SalePrice, fit_reg=True, scatter_kws={"alpha": 0.3})

    ax.set(yticklabels=[], xticklabels=[], xticks=[])

    plt.show()
for i in vars_:

    sns.jointplot(x=df_num[i], y=df_num.SalePrice, kind='kde')

    plt.show()
## Load package for linear regression 

import statsmodels.api as sm
df_num['stFrSF'] = df_num['1stFlrSF']

df_num['ndFrSF'] = df_num['2ndFlrSF']

df_num['SsnPorch'] = df_num['3SsnPorch']
## define num vars for the loop

num_vars = ['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',

           'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'LowQualFinSF', 

           'GrLivArea', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars',

           'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'ScreenPorch',

           'PoolArea', 'MiscVal', 'YrSold', 'stFrSF', 'ndFrSF', 'SsnPorch']
## fit each variable and save the R-squared in a list

num_rsquared = []

for var in num_vars:

    model = sm.OLS.from_formula(f'SalePrice ~ {var}', data=df_num)

    result = model.fit()

    num_rsquared.append((var,result.rsquared))

    

## make a series with rsquared data

rsquared = [i[1] for i in num_rsquared]

vars_ = [i[0] for i in num_rsquared]

df_num_rsquared = pd.Series(rsquared, index=vars_)



df_num_rsquared.sort_values(ascending=False)
## first define our categorical variables

cat_vars = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour',

       'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1',

       'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond',

       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',

       'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',

       'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC',

       'CentralAir', 'Electrical', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',

       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'Functional',

       'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',

       'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'MoSold', 'SaleType',

       'SaleCondition']
# load package for ANOVA

from statsmodels.formula.api import ols
## ANOVA this prints the number of variables with significant p-values

anova_pvalues = []

for var in cat_vars:

    model = ols(f'SalePrice ~ C({var})', data=df_cat).fit()

    anova_table = sm.stats.anova_lm(model, typ=2)

    pvalue = anova_table.loc[f'C({var})','PR(>F)']

    if pvalue < 0.05:

        anova_pvalues.append((var,pvalue))

    else:

        continue



print(len(anova_pvalues))
## fit each variable and save the R-squared in a list

cat_rsquared = []

for i in anova_pvalues:

    model = sm.OLS.from_formula(f'SalePrice ~ {i[0]}', data=df_cat)

    result = model.fit()

    cat_rsquared.append((i[0],result.rsquared))

    

## make a series with rsquared data

rsquared = [i[1] for i in cat_rsquared]

vars_ = [i[0] for i in cat_rsquared]

df_cat_rsquared = pd.Series(rsquared, index=vars_)



df_cat_rsquared.sort_values(ascending=False)
## the selected features and target

features = ['GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','TotRmsAbvGrd','YearBuilt',

            'YearRemodAdd','GarageYrBlt','MasVnrArea','OverallQual','Neighborhood','ExterQual','KitchenQual',

            'BsmtQual','FullBath','Alley','GarageFinish','Foundation']

X_test = test[features]

features.append('SalePrice')

X_train = train[features]
## Percentage of NAN values per column in train

NAN_train = [(column, X_train[column].isna().mean()*100) for column in list(X_train.columns)]

NAN_train = pd.DataFrame(NAN_train, columns=['column_name','percentage'])

NAN_train
## Percentage of NAN values per column in test

NAN_test = [(column, X_test[column].isna().mean()*100) for column in list(X_test.columns)]

NAN_test = pd.DataFrame(NAN_test, columns=['column_name','percentage'])

NAN_test
X_train = X_train.drop(['Alley'], axis=1)

X_test = X_test.drop(['Alley'], axis=1)



X_train['GarageYrBlt'].fillna(X_train.GarageYrBlt.median(), inplace=True)

X_train['MasVnrArea'].fillna(X_train.MasVnrArea.median(), inplace=True)

X_train.dropna(inplace=True)



X_test['TotalBsmtSF'].fillna(X_test.TotalBsmtSF.median(), inplace=True)

X_test['GarageYrBlt'].fillna(X_test.GarageYrBlt.median(), inplace=True)

X_test['MasVnrArea'].fillna(X_test.MasVnrArea.median(), inplace=True)

X_test['GarageArea'].fillna(X_test.GarageArea.median(), inplace=True)

X_test['GarageCars'].fillna(X_test.GarageCars.median(), inplace=True)
## separate SalePrice into a new target series and drop it from the train dataset

y = X_train['SalePrice']

X_train = X_train.drop(['SalePrice'], axis=1)
## get dummies

X_train = pd.get_dummies(X_train)

X_test = pd.get_dummies(X_test)
# Scale numeric data

from sklearn.preprocessing import StandardScaler



num_data = ['GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'TotRmsAbvGrd',

            'YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'MasVnrArea']

X_train[num_data] = StandardScaler().fit_transform(X_train[num_data])

X_test[num_data] = StandardScaler().fit_transform(X_test[num_data])
print(X_test.shape)

print(X_train.shape)
X_test.drop(X_test.columns.difference(list(X_train.columns)), axis=1, inplace=True)
print(X_test.shape)

print(X_train.shape)
## import cross_val for the scores

from sklearn.model_selection import cross_val_score
## Linear Regression

from sklearn.linear_model import LinearRegression



linreg = LinearRegression()

cv = cross_val_score(linreg, X_train, y, cv=5)

print(cv)

print(cv.mean())
## k neighbors regressor selecting number of neighbors

from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_predict



error = []

for k in range(1,51):

    knn = KNeighborsRegressor(n_neighbors=k)

    y_pred = cross_val_predict(knn, X_train, y, cv=5)

    error.append(mean_squared_error(y,y_pred))

    

plt.plot(range(1,51), error);
## look only between 5 and 20

error = []

for k in range(5,21):

    knn = KNeighborsRegressor(n_neighbors=k)

    y_pred = cross_val_predict(knn, X_train, y, cv=5)

    error.append(mean_squared_error(y,y_pred))

    

plt.plot(range(5,21), error);
## k neighbors regressor

## smallest error  seems to be at 11 neighbors

knnreg = KNeighborsRegressor(n_neighbors = 11)

cv = cross_val_score(knnreg, X_train, y, cv=5)

print(cv)

print(cv.mean())
## Ridge regression

## looking at different alpha values

from sklearn.linear_model import Ridge



error = []

for alpha in range(1,51):

    linridge = Ridge(alpha=float(alpha))

    y_pred = cross_val_predict(linridge, X_train, y, cv=5)

    error.append(mean_squared_error(y,y_pred))

    

plt.plot(range(1,51), error);
## Ridge regression

## smallest error at alpha=1

linridge = Ridge(alpha=1.0)

cv = cross_val_score(linridge, X_train, y, cv=5)

print(cv)

print(cv.mean())
## Lasso regression

## looking at different alpha values

from sklearn.linear_model import Lasso



error = []

for alpha in range(1,51):

    linlasso = Lasso(alpha=float(alpha), max_iter = 10000)

    y_pred = cross_val_predict(linlasso, X_train, y, cv=5)

    error.append(mean_squared_error(y,y_pred))

    

plt.plot(range(1,51), error);
## Lasso regression

## looking at different alpha values

## looking only between 3 and 12

from sklearn.linear_model import Lasso



error = []

for alpha in range(3,13):

    linlasso = Lasso(alpha=float(alpha), max_iter = 10000)

    y_pred = cross_val_predict(linlasso, X_train, y, cv=5)

    error.append(mean_squared_error(y,y_pred))

    

plt.plot(range(3,13), error);
## Lasso regression

## smallest error at alpha = 9

linlasso = Lasso(alpha=9.0)

cv = cross_val_score(linlasso, X_train, y, cv=5)

print(cv)

print(cv.mean())
## Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor



rf = RandomForestRegressor(random_state=0, n_estimators=200, max_depth=10)

cv = cross_val_score(rf, X_train, y, cv=5)

print(cv)

print(cv.mean())
## Gradient Boosting Regressor

from sklearn.ensemble import GradientBoostingRegressor



gb = GradientBoostingRegressor(random_state=0, n_estimators=200, max_depth=3)

cv = cross_val_score(gb, X_train, y, cv=5)

print(cv)

print(cv.mean())
## XGB Regressor

from xgboost import XGBRegressor



xgb = XGBRegressor(random_state=0, n_estimators=30, max_depth=3)

cv = cross_val_score(xgb, X_train, y, cv=5)

print(cv)

print(cv.mean())
## LGBM Regressor

from lightgbm import LGBMRegressor



lgbm = LGBMRegressor(random_state=0, n_estimators=100, max_depth=3)

cv = cross_val_score(lgbm, X_train, y, cv=5)

print(cv)

print(cv.mean())
gb.fit(X_train, y)

predictions = gb.predict(X_test)
## make the file to submit

df_submition = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')



df_submition['SalePrice'] = predictions



df_submition.drop(df_submition.columns.difference(['Id', 'SalePrice']), axis=1, inplace=True) # Selecting only needed columns



print(df_submition.shape)



df_submition.head(5) 
df_submition.to_csv('my_submission.csv', index=False)

print('File created')