# The Packages we need for this project:

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy

import warnings

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, Ridge, Lasso, SGDRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm, gaussian_process
from sklearn.metrics import mean_squared_error


# The basic setup for this project:

sns.set(style="ticks")
%matplotlib inline
warnings.filterwarnings('ignore')


# The data we need for this project:

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train.head()
train.shape, test.shape
train.info()
test.info()
# Describe and visualize the dependent variables -- house prices:

print(train.shape)
train['SalePrice'].describe()
sns.set(rc = {'figure.figsize':(12,7)})
sns.distplot(train['SalePrice'], rug=True)
# Have a look at log of price to promise the dependent variables can obey normal distribution which don't change the veracity of final result at the sametime:

train['LogPrice'] = np.log(train['SalePrice'])
sns.distplot(train['LogPrice'], rug=True)
# Use correlation matrix to describe the relationship among all features:

corrmat = train.corr()
sns.set(rc = {'figure.figsize':(35,15)})
sns.heatmap(corrmat, cmap="YlGnBu", annot=True)
# The correlation matrix among selected features:

cols = ['LogPrice', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea']
corrmat = train[cols].corr()
sns.set(rc = {'figure.figsize':(12,7)})
sns.heatmap(corrmat, cmap="YlGnBu", annot=True)
# Decribe every single variables:

print(train['OverallQual'].describe())
sns.set(rc = {'figure.figsize':(12,7)})
sns.countplot(train['OverallQual'])
print(train['YearBuilt'].describe())
sns.set(rc = {'figure.figsize':(35,15)})
p = sns.countplot(train['YearBuilt'])
p.set_xticklabels(p.get_xticklabels(), rotation=30)
print(train['YearRemodAdd'].describe())
sns.countplot(train['YearRemodAdd'])
print(train['TotalBsmtSF'].describe())
sns.set(rc = {'figure.figsize':(12,7)})
sns.distplot(train['TotalBsmtSF'], rug=True)
print(train['1stFlrSF'].describe())
sns.distplot(train['1stFlrSF'], rug=True)
print(train['GrLivArea'].describe())
sns.distplot(train['GrLivArea'], rug=True)
print(train['FullBath'].describe())
sns.countplot(train['FullBath'])
print(train['TotRmsAbvGrd'].describe())
sns.countplot(train['TotRmsAbvGrd'])
print(train['GarageCars'].describe())
sns.countplot(train['GarageCars'])
print(train['GarageArea'].describe())
sns.distplot(train['GarageArea'], rug=True)
# Find the missing data:

train_data = train[cols]
missing_data = train_data.isnull().sum().sort_values(ascending=False)
missing_percent = (train_data.isnull().sum()*100/train_data.isnull().count()).sort_values(ascending=False)
pd.concat([missing_data, missing_percent], axis=1, keys=['missing_data', 'missing_percent'])
# Standardize varibles with log function:

train_data['LogGarageArea'] = train_data['GarageArea']
train_data.loc[train_data['LogGarageArea']>0, 'LogGarageArea'] = np.log(train_data['LogGarageArea'])
sns.distplot(train_data[train_data['LogGarageArea']>0]['LogGarageArea'], rug=True)
train_data['LogGrLivArea'] = np.log(train_data['GrLivArea'])
sns.distplot(train_data['LogGrLivArea'], rug=True)
train_data['Log1stFlrSF'] = np.log(train_data['1stFlrSF'])
sns.distplot(train_data['Log1stFlrSF'], rug=True)
train_data['LogTotalBsmtSF'] = train_data['TotalBsmtSF']
train_data.loc[train_data['LogTotalBsmtSF']>0, 'LogTotalBsmtSF'] = np.log(train_data['LogTotalBsmtSF'])
sns.distplot(train_data[train_data['LogTotalBsmtSF']>0]['LogTotalBsmtSF'], rug=True)
train_data.head(10)
# The quantitative variables:

columns_quantity = ['LogTotalBsmtSF', 'Log1stFlrSF', 'LogGrLivArea', 'LogGarageArea']
# A positive correlation between 'LogTotalBsmtSF' and 'LogPrice':

sns.scatterplot(x=train_data['LogTotalBsmtSF'], y=train_data['LogPrice'])
# A positive correlation between 'Log1stFlrSF' and 'LogPrice':

sns.scatterplot(x=train_data['Log1stFlrSF'], y=train_data['LogPrice'])
# A positive correlation between 'LogGrLivArea' and 'LogPrice':

sns.scatterplot(x=train_data['LogGrLivArea'], y=train_data['LogPrice'])
# A positive correlation between 'LogGarageArea' and 'LogPrice':

sns.scatterplot(x=train_data['LogGarageArea'], y=train_data['LogPrice'])
# The categorical variable:

colums_cate = ['OverallQual', 'YearBuilt', 'YearRemodAdd', 'FullBath', 'GarageCars', 'TotRmsAbvGrd']
# A positive correlation between 'OverallQual' and 'LogPrice':

sns.boxplot(train_data['OverallQual'], train_data['LogPrice'])
# A positive correlation between 'FullBath' and 'LogPrice':

sns.boxplot(train_data['FullBath'], train_data['LogPrice'])
# A positive correlation between 'GarageCars' and 'LogPrice' but it turns negative when 'GarageCars' is too large:

sns.boxplot(train_data['GarageCars'], train_data['LogPrice'])
# A positive correlation between 'TotRmsAbvGrd' and 'LogPrice' but it turns negative when 'TotRmsAbvGrd' is too large:

sns.boxplot(train_data['TotRmsAbvGrd'], train_data['LogPrice'])
# A positive correlation tendency between 'YearBuilt' and 'LogPrice' :

sns.set(rc = {'figure.figsize':(35,15)})
p = sns.boxplot(train_data['YearBuilt'], train_data['LogPrice'])
p.set_xticklabels(p.get_xticklabels(), rotation=30)
# A positive correlation tendency between 'YearRemodAdd' and 'LogPrice' 

p = sns.boxplot(train_data['YearRemodAdd'], train_data['LogPrice'])
p.set_xticklabels(p.get_xticklabels(), rotation=30)
train_data.loc[train_data['YearRemodAdd']==train_data['YearBuilt'], 'RemodAdd'] = 0
train_data['RemodAdd'].fillna(1, inplace=True)
p = sns.boxplot(hue=train_data['RemodAdd'], x=train_data['YearBuilt'], y=train_data['LogPrice'])
p.set_xticklabels(p.get_xticklabels(), rotation=30)
cols = ['LogPrice', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'FullBath', 'GarageCars', 'TotRmsAbvGrd', 'LogTotalBsmtSF', 'Log1stFlrSF', 'LogGrLivArea', 'LogGarageArea']
corrm = train_data[cols].corr()
sns.set(rc = {'figure.figsize':(12,7)})
sns.heatmap(corrm, cmap="YlGnBu", annot=True)
cols = ['LogPrice', 'OverallQual', 'YearBuilt', 'RemodAdd', 'Log1stFlrSF', 'FullBath', 'LogGrLivArea', 'GarageCars']
sns.pairplot(train_data[cols])
# Data preparation:

train_data['Cross'] = train_data['YearBuilt'] * train_data['RemodAdd']
cols = ['LogPrice', 'OverallQual', 'YearBuilt', 'Cross', 'Log1stFlrSF', 'FullBath', 'LogGrLivArea', 'GarageCars']
X = train_data[cols[1:]]
Y = train_data[cols[0]]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)


# Model Score:

def r_square(model, x, y):
    print('R^2 is {}'.format(model.score(x, y)))
    return model.score(x, y)
    
def rmse(y_true, y_pred):
    print('RMSE is {}'.format(mean_squared_error(y_true, y_pred)))
    return mean_squared_error(y_true, y_pred)

sns.set(rc = {'figure.figsize':(12,7)})
# Linear Regression with Ordinary Least Squares:

lrm = LinearRegression().fit(x_train, y_train)
coe1 = pd.DataFrame(list(zip(X.columns, lrm.coef_)), columns=['Variables', 'Coefficients'])
print(coe1)

y_train_pred = lrm.predict(x_test)

r_square(lrm, x_test, y_test)
rmse(y_test, y_train_pred)

plt.scatter(y_test, y_train_pred, alpha=0.75)
plt.xlabel('y_true'); plt.ylabel('y_predict')

# Linear Regression with Ridge Regression:

alphas_rr=[0.0001, 0.0003, 0.0005, 0.0007, 0.0009, 0.01, 0.05, 0.1, 0.5, 1, 3, 5, 10, 30, 50, 100]
r_squares = []
rmses = []

for alpha in alphas_rr:
    
    rrm = Ridge(alpha).fit(x_train, y_train)
    coe2 = pd.DataFrame(list(zip(X.columns, rrm.coef_)), columns=['Variables', 'Coefficients'])
    print(coe2)

    y_train_pred = rrm.predict(x_test)

    print('alpha is {}'.format(alpha))
    r_squares.append(r_square(rrm, x_test, y_test))
    rmses.append(rmse(y_test, y_train_pred))

    plt.scatter(y_test, y_train_pred, alpha=0.75)
    plt.xlabel('y_true'); plt.ylabel('y_predict')
    plt.show()
    print()

plt.plot(alphas_rr, rmses)
# Linear Regression with Lasso Regression:

alphas_lar=[0.0001, 0.0003, 0.0005, 0.0007, 0.0009, 0.01, 0.05, 0.1]
r_squares = []
rmses = []

for alpha in alphas_lar:
    
    larm = Lasso(alpha).fit(x_train, y_train)
    coe2 = pd.DataFrame(list(zip(X.columns, larm.coef_)), columns=['Variables', 'Coefficients'])
    print(coe2)

    y_train_pred = larm.predict(x_test)

    print('alpha is {}'.format(alpha))
    r_squares.append(r_square(rrm, x_test, y_test))
    rmses.append(rmse(y_test, y_train_pred))

    plt.scatter(y_test, y_train_pred, alpha=0.75)
    plt.xlabel('y_true'); plt.ylabel('y_predict')
    plt.show()
    print()
    
plt.plot(alphas_lar, rmses)
# Linear Regression with XGBoost:

xgbr = XGBRegressor(max_depth=5, learning_rate=0.01, n_estimators=1460, objective='reg:linear').fit(x_train, y_train)

y_train_pred = xgbr.predict(x_test)

r_square(xgbr, x_test, y_test)
rmse(y_test, y_train_pred)

plt.scatter(y_test, y_train_pred, alpha=0.75)
plt.xlabel('y_true'); plt.ylabel('y_predict')

# Use Linear Regression with XGBoost to predict data:

# Data preparation:
test['Log1stFlrSF'] = np.log(test['1stFlrSF'])
test['LogGrLivArea'] = np.log(test['GrLivArea'])
test.loc[test['YearRemodAdd']==test['YearBuilt'], 'RemodAdd'] = 0
test['RemodAdd'].fillna(1, inplace=True)
test['Cross'] = test['YearBuilt'] * test['RemodAdd']

final_cols = ['OverallQual', 'YearBuilt', 'Cross', 'Log1stFlrSF', 'FullBath', 'LogGrLivArea', 'GarageCars']

test_data = test[final_cols]
test_data.head()

# Model creation and data predition:

model = XGBRegressor(max_depth=5, learning_rate=0.01, n_estimators=1460, objective='reg:linear').fit(X, Y)

y_test_pred = model.predict(test_data)
print('Initiate Predictions are {}'.format(y_test_pred))

final_predictions = np.exp(y_test_pred)
print('Final Predictions are {}'.format(final_predictions))

# Form result:

submission = pd.DataFrame({'Id':test['Id'], 'SalePrice':final_predictions})
submission.head()
# Data submission:

submission.to_csv('submission.csv', index=False)