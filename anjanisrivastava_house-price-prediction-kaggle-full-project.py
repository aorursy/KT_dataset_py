# Importing necessary Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
pd.options.display.max_columns = None
pd.options.display.max_rows = None
#Importing the datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train.head()
#Checking the first five columns of data
test.head()
#Checking the shape of both training and test datasets so as to get the no. of rows and columns
print("Training Set: ", train.shape)
print("Test Set: ", test.shape)
#Getting the collinearity between variables
train.corr()
'''First taking a glance at the Target variable 'SalePrice'. We should make sure that it is normally distributed.
But, from the Graph , It is right skewed'''
plt.figure(figsize=(12,8))
sns.distplot(train['SalePrice'], kde=True, color='red')
#This makes the target variable somewhere normally distributed
plt.figure(figsize=(12,8))
log_value = np.log1p(train['SalePrice'])
sns.distplot(log_value)
#Checking the skewness and kurtosis of target variable
print("Skewness: " + str(train['SalePrice'].skew()))
print("Kurtosis: " + str(train['SalePrice'].kurt()))
#Multivariate plot
col = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'GarageArea']
sns.set(style='ticks')
sns.pairplot(train[col], height=8, kind='reg')
#Checking highly correalated variables relational plot with the target variable
plt.figure(figsize=(10,6))
sns.boxplot(x=train['OverallQual'], y=train['SalePrice'], data=train)
plt.figure(figsize=(10,6))
sns.scatterplot(x=train['GrLivArea'], y=train['SalePrice'], data=train)
plt.figure(figsize=(10,6))
sns.boxplot(x=train['GarageCars'], y=train['SalePrice'], data=train)
plt.figure(figsize=(10,6))
sns.scatterplot(x=train['GarageArea'], y=train['SalePrice'], data=train)
plt.figure(figsize=(10,6))
sns.scatterplot(x=train['TotalBsmtSF'], y=train['SalePrice'], data=train)
plt.figure(figsize=(10,6))
sns.scatterplot(x=train['1stFlrSF'], y=train['SalePrice'], data=train)
plt.figure(figsize=(10,6))
sns.boxplot(x=train['FullBath'], y=train['SalePrice'], data=train)
plt.figure(figsize=(10,6))
sns.scatterplot(x=train['YearBuilt'], y=train['SalePrice'], data=train)
plt.figure(figsize=(10,6))
sns.scatterplot(x=train['YearRemodAdd'], y=train['SalePrice'], data=train)
plt.figure(figsize=(10,6))
sns.scatterplot(x=train['GarageYrBlt'], y=train['SalePrice'], data=train)
plt.figure(figsize=(10,6))
sns.scatterplot(x=train['GrLivArea'], y=train['SalePrice'], data=train)
plt.figure(figsize=(10,6))
sns.boxplot(x=train['TotRmsAbvGrd'], y=train['SalePrice'], data=train)
#Dropping the columns having more NaNs
train = train.drop(['Alley', 'PoolQC', 'MiscFeature'], axis=1)
test = test.drop(['Alley', 'PoolQC', 'MiscVal'], axis=1)
print("Training Set: ", train.shape)
print("Test Set: ", test.shape)
# Checking that what type of data variables we have
train_corr = train.select_dtypes(include=[np.number])
train_corr.shape
test_corr = test.select_dtypes(include=[np.number])
test_corr.shape
train_corr = train.select_dtypes(include=[object])
train_corr.shape
test_corr = test.select_dtypes(include=[object])
test_corr.shape
#Storing the ID values as we have to use it when we make a submission
Id_values = test.Id
#Deleting unnecessary variables
del train['Id']
del test['Id']
del train['Utilities']
del test['Utilities']
print("Training Set: ", train.shape)
print("Test Set: ", test.shape)
## Deleting those two values with outliers. 
train = train[train.GrLivArea < 4500]
train.reset_index(drop = True, inplace = True)
## Deleting those two values with outliers. 
train = train[train.TotalBsmtSF < 4000]
train.reset_index(drop = True, inplace = True)
print("Training Set: ", train.shape)
print("Test Set: ", test.shape)
#Filling up NaNs
train['Fence'] = train['Fence'].fillna('None')
train['FireplaceQu'] = train['FireplaceQu'].fillna('None')
test['Fence'] = test['Fence'].fillna('None')
test['FireplaceQu'] = test['FireplaceQu'].fillna('None')
#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
train['LotFrontage'] = train.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
test['LotFrontage'] = test.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    train[col] = train[col].fillna('None')
    test[col] = test[col].fillna('None')
for col in ['GarageYrBlt', 'GarageArea', 'GarageCars']:
    train[col] = train[col].fillna(int(0))
    test[col] = test[col].fillna(int(0))
for col in ('BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtCond', 'BsmtQual'):
    train[col] = train[col].fillna('None')
    test[col] = test[col].fillna('None')
train['MasVnrArea'] = train['MasVnrArea'].fillna(int(0))
train['MasVnrType'] = train['MasVnrType'].fillna('None')
train['Electrical'] = train['Electrical'].fillna(train['Electrical']).mode()[0]
test['MasVnrArea'] = test['MasVnrArea'].fillna(int(0))
test['MasVnrType'] = test['MasVnrType'].fillna('None')
test['Electrical'] = test['Electrical'].fillna(train['Electrical']).mode()[0]
#train = train.drop(['Utilities'], axis=1)
#test = test.drop(['Utilities'], axis=1)
for col in ['MSZoning', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'Functional', 'MiscFeature']:
    test[col] = test[col].fillna('None')
test['BsmtFinSF1'] = test['BsmtFinSF1'].fillna(train['BsmtFinSF1']).mode()[0]
test['BsmtFinSF2'] = test['BsmtFinSF2'].fillna(train['BsmtFinSF2']).mode()[0]
test['BsmtUnfSF'] = test['BsmtUnfSF'].fillna(train['BsmtUnfSF']).mode()[0]
test['TotalBsmtSF'] = test['TotalBsmtSF'].fillna(train['TotalBsmtSF']).mode()[0]
test['BsmtFullBath'] = test['BsmtFullBath'].fillna(train['BsmtFullBath']).mode()[0]
test['BsmtHalfBath'] = test['BsmtHalfBath'].fillna(train['BsmtHalfBath']).mode()[0]
test['SaleType'] = test['SaleType'].fillna(train['SaleType']).mode()[0]

#Checking th
test.isnull().sum()
train.isnull().sum()
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold', 'MSZoning', 'LandContour', 'LotConfig', 'Neighborhood',
        'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
        'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'Foundation', 'GarageType', 
        'SaleType', 'SaleCondition', 'Electrical', 'Heating')
from sklearn.preprocessing import LabelEncoder
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(train[c].values)) 
    train[c] = lbl.transform(list(train[c].values))
from sklearn.preprocessing import LabelEncoder
for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(test[c].values)) 
    test[c] = lbl.transform(list(test[c].values))
train.shape
test.shape
test.head()
test.info()
train.info()

# Store target variable of training data in a safe place
sale_price = train.SalePrice
del train['SalePrice']
print("Training Set: ", train.shape)
print("Test Set: ", test.shape)
# Concatenate training and test sets
data = pd.concat([train,test], sort=True)
data.info()
del data['MiscFeature']
del data['MiscVal']
data.info()
data_train = data.iloc[:1458]
data_test = data.iloc[1458:]
data_train.info()
data_test.info()
X = data_train.values
test = data_test.values
y = sale_price.values
from sklearn.ensemble import GradientBoostingRegressor
GBR = GradientBoostingRegressor(n_estimators=100, max_depth=4)
X.shape[0]
y.shape[0]
GBR.fit(X, y)
# Make predictions and store in 'Survived' column of df_test
Y_pred = GBR.predict(test)
Y_pred
Y_pred.shape
x_value = np.arange(start=1461, stop=2920)
x_value
x_value.shape
dta = pd.DataFrame({'Id':x_value, 'SalePrice':Y_pred})
dta.head()
dta.to_csv('submit3csv', index=False)

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(X, y)
predictions = rf.predict(test)
x_val = np.arange(start=1461, stop=2920)
dta1 = pd.DataFrame({'Id':x_val, 'SalePrice':predictions})
dta.to_csv('submit4', index=False)




















































