import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O
pd.set_option('display.max_columns', None)
import seaborn as sns # Visualization
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats
from scipy.stats import norm
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
# Checking for outliers in Y variable
train['SalePrice'].describe()
# Distribution
sns.distplot((train['SalePrice']))
train_wo_Y = train.loc[:,train.columns != 'SalePrice']
data = train_wo_Y.append(test)
data.head()
# Finding correlation between variables
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(data.corr(), vmax=.8, square=True);
# Removing variables with high collinearity
Variables_not_req = ['GarageYrBlt','TotalBsmtSF','TotRmsAbvGrd','GarageCars']
data = data.drop(Variables_not_req,axis = 1)
data
# Handling missing values
missing = data.isnull().sum()
total = data.isnull().count()
missing_data = pd.concat([missing,total],axis = 1, keys = ['# of missing','Total'])
missing_data['% missing'] = missing_data['# of missing'] / missing_data['Total']
missing_data = missing_data.sort_values(by= '% missing', ascending = False).head(35)
missing_data
Variables_w_15p_missing_values = missing_data[missing_data['% missing'] >= 0.15].index
data = data.drop(Variables_w_15p_missing_values,axis = 1)
Variables_w_missing_values = missing_data[missing_data['% missing'] < 0.15]
Variables_w_missing_values = Variables_w_missing_values[Variables_w_missing_values['% missing'] > 0]

Variables_w_missing_values
# Let's see the distribution of numbers before imputing
data['GarageQual'] = data['GarageQual'].fillna("NA")
data['GarageCond'] = data['GarageCond'].fillna("NA")
data['GarageFinish'] = data['GarageFinish'].fillna("NA")
data['GarageType'] = data['GarageType'].fillna("NA")
# As similar to garage, Basement also has na due to No basement
data['BsmtExposure'] = data['BsmtExposure'].fillna("NA")
data['BsmtCond'] = data['BsmtCond'].fillna("NA")
data['BsmtQual'] = data['BsmtQual'].fillna("NA")
data['BsmtFinType2'] = data['BsmtFinType2'].fillna("NA")
data['BsmtFinType1'] = data['BsmtFinType1'].fillna("NA")
# Since mode of 'Masonry veneer type' is None-1742. We will replace NA with None and Make area to 0. 
data['MasVnrType'] = data['MasVnrType'].fillna("None")
data['MasVnrArea'] = data['MasVnrType'].fillna(0)
# MSZoning - Since only 4(~ 0.1%) are missing, we will impuute it with mode
# Imputing with mode for Basement bath, utilities and functinal
## data['MSZoning'].describe()
data['MSZoning'] = data['MSZoning'].fillna("RL")
data['BsmtFullBath'] = data['BsmtFullBath'].fillna(0)
data['BsmtHalfBath'] = data['BsmtHalfBath'].fillna(0)
data['Utilities'] = data['Utilities'].fillna("AllPub")
data['Functional'] = data['Functional'].fillna("Typ")
# For imputing Basement Finished sq, we check basement Finish type
print(data[data['BsmtFinSF1'].isnull()].loc[:,['BsmtFinType1','BsmtFinSF1']])
print(data[data['BsmtFinSF2'].isnull()].loc[:,['BsmtFinType2','BsmtFinSF2']])
# Since No basement imputing with 0
data['BsmtFinSF1'] = data['BsmtFinSF1'].fillna(0)
data['BsmtFinSF2'] = data['BsmtFinSF2'].fillna(0)

# We don't have any information for Unfinished basement, we are imputing with mode
data['BsmtUnfSF'] = data['BsmtUnfSF'].fillna(0)

# Filling with mode
data['KitchenQual'] = data['KitchenQual'].fillna(data['KitchenQual'].mode()[0])
data['Electrical'] = data['Electrical'].fillna(data['Electrical'].mode()[0])
data['Exterior1st'] = data['Exterior1st'].fillna(data['Exterior1st'].mode()[0])
data['Exterior2nd'] = data['Exterior2nd'].fillna(data['Exterior2nd'].mode()[0])
data['SaleType'] = data['SaleType'].fillna(data['SaleType'].mode()[0])
data['GarageArea'] = data['GarageArea'].fillna(data['GarageArea'].mode()[0])
# Checking if all values has been imputed
data.isnull().sum().sort_values(ascending = False).head(3)
# Removing Id column
data = data.drop(['Id'],axis = 1)
# Converting Categorical columns into dummy variable
data = pd.get_dummies(data)
# Splitting the Test and train data after Cleaning
train_cleaned = data.head(1460)
test_cleaned = data.tail(1459)
# Correlation between sale price and other variables
train_df = pd.concat([train_cleaned,train['SalePrice']],axis = 1)
train_df.corr()[['SalePrice']].sort_values(by = 'SalePrice', ascending = False).head(25)

# plt.figure(figsize=(5,20))
# sns.heatmap(train.corr()[['SalePrice']].sort_values(by=['SalePrice'],ascending=False).head(50), vmin=-1, annot=True);
req_var = ['KitchenQual_TA','ExterQual_TA','OverallQual','GrLivArea','GarageArea',
            '1stFlrSF','FullBath','BsmtQual_Ex','YearBuilt','KitchenQual_Ex']

train_cleaned = train_cleaned.loc[:,req_var]
test_cleaned = test_cleaned.loc[:,req_var]
train_cleaned = pd.concat([train['SalePrice'],train_cleaned],axis = 1)
train_cleaned
# Checking for linearity by plotting pair plot

sns.set()
sns.pairplot(data = train_cleaned, 
             y_vars = ['SalePrice'], 
             x_vars = ['KitchenQual_TA','ExterQual_TA','OverallQual','GrLivArea', 'GarageArea','1stFlrSF','FullBath','BsmtQual_Ex','YearBuilt','KitchenQual_Ex'] )
plt.show()

# pp = sns.pairplot(data=data,
#                   y_vars=['age'],
#                   x_vars=['weight', 'height', 'happiness'])
# 'OverallQual','GrLivArea', 'GarageArea','1stFlrSF'
sns.distplot((train_cleaned['GrLivArea']), fit=norm);
fig = plt.figure()
res = stats.probplot((train_cleaned['GrLivArea']),dist = norm, plot=plt)
# 'OverallQual','GrLivArea', 'GarageArea','1stFlrSF'
train_cleaned['GrLivArea'] = np.log(train_cleaned['GrLivArea'])
test_cleaned['GrLivArea'] = np.log(test_cleaned['GrLivArea'])

sns.distplot((a['GrLivArea']), fit=norm);
fig = plt.figure()
res = stats.probplot((a['GrLivArea']),dist = norm, plot=plt)
# 'OverallQual','GrLivArea', 'GarageArea','1stFlrSF'
sns.distplot((train_cleaned['GarageArea']), fit=norm);
fig = plt.figure()
res = stats.probplot((train_cleaned['GarageArea']),dist = norm, plot=plt)

# Finding Non-zero minimum number in GarageArea
pd.DataFrame(train_cleaned['GarageArea'].unique()).sort_values(by = 0)
# Replacing 0 with minimum value/2
train_cleaned['GarageArea'] = train_cleaned['GarageArea'].replace(0,80)
test_cleaned['GarageArea'] = test_cleaned['GarageArea'].replace(0,80)

train_cleaned['GarageArea'] = np.sqrt(train_cleaned['GarageArea'])
test_cleaned['GarageArea'] = np.sqrt(test_cleaned['GarageArea'])

sns.distplot(train_cleaned['GarageArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(train_cleaned['GarageArea'],dist = norm, plot=plt)
# 'OverallQual','GrLivArea', 'GarageArea','1stFlrSF'
sns.distplot((train_cleaned['1stFlrSF']), fit=norm);
fig = plt.figure()
res = stats.probplot((train_cleaned['1stFlrSF']),dist = norm, plot=plt)

# a['KitchenQual_Ex'].describe()
train_cleaned['1stFlrSF'] = np.log(train_cleaned['1stFlrSF'])
test_cleaned['1stFlrSF'] = np.log(test_cleaned['1stFlrSF'])

sns.distplot((train_cleaned['1stFlrSF']), fit=norm);
fig = plt.figure()
res = stats.probplot((train_cleaned['1stFlrSF']),dist = norm, plot=plt)
train_cleaned['SalePrice'] = np.log(train_cleaned['SalePrice'])

sns.distplot((train_cleaned['SalePrice']), fit=norm);
fig = plt.figure()
res = stats.probplot((train_cleaned['SalePrice']),dist = norm, plot=plt)

# import scipy.stats as stats
# stats.probplot(x1[:,0], dist="norm", plot=plt)
# plt.show()
sns.set()
sns.pairplot(data = train_cleaned, 
             y_vars = ['SalePrice'], 
             x_vars = ['KitchenQual_TA','ExterQual_TA','OverallQual','GrLivArea', 'GarageArea','1stFlrSF','FullBath','BsmtQual_Ex','YearBuilt','KitchenQual_Ex'] )
plt.show()
# Create linear regression
regr = LinearRegression()

# Fit the linear regression
# model = regr.fit(train_cleaned, np.log(train['SalePrice']))
# model = regr.fit(a, (a['SalePrice']))

train_x = train_cleaned.drop(['SalePrice'],axis = 1)
train_y = train_cleaned['SalePrice']

model = regr.fit(train_x,train_y)
pred = pd.DataFrame(np.exp(model.predict(test_cleaned)))
Id = pd.DataFrame(test['Id'])
# submit 
submit = pd.concat([Id,pred],axis = 1)
submit = submit.rename(columns = {0:"SalePrice"})
submit.to_csv('submission.csv', index=False)

import statsmodels.formula.api as sm
model = sm.ols(formula='SalePrice ~ KitchenQual_TA + ExterQual_TA + OverallQual + GrLivArea + GarageArea + FullBath + BsmtQual_Ex + YearBuilt + KitchenQual_Ex', data=train_cleaned)
# KitchenQual_TA	ExterQual_TA	OverallQual	GrLivArea	GarageArea	1stFlrSF	FullBath	BsmtQual_Ex	YearBuilt	KitchenQual_Ex
# model = sm.ols(formula='SalePrice ~ OverallQual + GrLivArea + GarageArea + BsmtQual_Ex + YearBuilt + KitchenQual_Ex', data=a)
# model = sm.ols(formula='np.log(SalePrice) ~ KitchenQual_TA + OverallQual + GrLivArea + GarageArea + BsmtQual_Ex + YearBuilt + KitchenQual_Ex', data=a)

fitted1.summary()
fitted1 = model.fit()
pred = np.exp(fitted1.predict(test_cleaned))
Id = pd.DataFrame(test['Id'])

submit = pd.concat([Id,pred],axis = 1)
submit = submit.rename(columns = {0:"SalePrice"})
submit.to_csv('submission.csv', index=False)
# # Finding R square

# import statsmodels.api as sma
# X_train = sma.add_constant(x_train) ## let's add an intercept (beta_0) to our model
# X_test = sma.add_constant(x_test) 

# # Linear regression can be run by using sm.OLS:
# import statsmodels.formula.api as sm
# lm2 = sm.OLS(y_train,X_train).fit()
# The summary of our model can be obtained via:
# lm2.summary()
print(model.score(train_x, train_y))
# model.coef_
# model.summary()

import statsmodels.api as sm

mod = sm.OLS(train['SalePrice'],train_cleaned)

fii = mod.fit()

p_values = fii.summary2().tables[1]['P>|t|']
# p_values.sort_values(ascending = False)
print(p_values)
print(model.coef_)
print(model.intercept_)
