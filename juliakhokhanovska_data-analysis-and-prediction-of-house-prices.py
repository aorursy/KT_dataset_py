import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set_style('whitegrid')
#As dataset has a lot of columns, let's change the default display options
#Doing that we will have full picture of our dataframe structure
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
houses_train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', index_col = 0)
houses_test_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', index_col = 0)
houses_train_data.head()
houses_train_data.info()
houses_test_data.head()
houses_test_data.info()
houses_train_data.isnull().sum().sort_values(ascending = False)
houses_test_data.isnull().sum().sort_values(ascending = False)
# Checking correlation between LotFrontage and LotArea.

houses_train_data[['LotFrontage', 'LotArea']].corr()
# Checking correlation between Garage Cars and Garage Area.

houses_train_data[['GarageCars', 'GarageArea']].corr()
houses_test_data[houses_test_data['GarageCars'].isnull()]
houses_test_data['GarageCars'].value_counts()
pd.get_dummies(houses_train_data[['SalePrice', 'BsmtCond', 'BsmtExposure', 'BsmtQual', 
                                  'BsmtFinType1', 'BsmtFinType2', 'BsmtHalfBath', 'BsmtFullBath', 
                                  'BsmtFinSF2', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF']]).corr()
# Now let's check how 'MasVnrType' and 'MasVnrArea' affect sales price. 

pd.get_dummies(houses_train_data[['SalePrice', 'MasVnrArea', 'MasVnrType']]).corr()
pd.get_dummies(houses_train_data[['SalePrice', 'OverallQual', 'OverallCond', 'ExterQual', 'ExterCond',
                                  'BsmtQual', 'BsmtCond', 'BsmtExposure', 'KitchenQual', 
                                  'SaleCondition' ]]).corr()
pd.get_dummies(houses_train_data[['SalePrice', 'Electrical', 'MSZoning', 'Utilities', 
                                  'Functional', 'SaleType', 'Exterior1st', 'Exterior2nd']]).corr()
# Checking Year-Price dependency
print((houses_train_data[['YearBuilt', 'SalePrice']].corr()))
sns.scatterplot(houses_train_data['YearBuilt'], houses_train_data['SalePrice'])
#dropping columns, where most of parameters are missing or weak correlation was detected
houses_train_data.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu', 'LotFrontage', 
                        'GarageType', 'GarageCond', 'GarageFinish', 'GarageQual', 'GarageYrBlt', 
                        'GarageArea', 'BsmtCond', 'BsmtFinType1', 'BsmtFinType2', 'BsmtHalfBath', 
                        'BsmtFullBath', 'BsmtFinSF2', 'BsmtFinSF1', 'BsmtUnfSF', 'MasVnrType', 
                        'OverallCond','ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 
                        'KitchenQual', 'SaleCondition', 'Electrical', 'MSZoning', 'Utilities', 
                        'Functional', 'SaleType', 'Exterior1st', 'Exterior2nd'], axis = 1, inplace = True)
houses_test_data.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu', 'LotFrontage', 
                        'GarageType', 'GarageCond', 'GarageFinish', 'GarageQual', 'GarageYrBlt', 
                        'GarageArea', 'BsmtCond', 'BsmtFinType1', 'BsmtFinType2', 'BsmtHalfBath', 
                        'BsmtFullBath', 'BsmtFinSF2', 'BsmtFinSF1', 'BsmtUnfSF', 'MasVnrType', 
                        'OverallCond','ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 
                        'KitchenQual', 'SaleCondition', 'Electrical', 'MSZoning', 'Utilities', 
                        'Functional', 'SaleType', 'Exterior1st', 'Exterior2nd'], axis = 1, inplace = True)

#filling the gaps accordingly to notes above
houses_train_data.fillna({'TotalBsmtSF': 0, 'MasVnrArea': houses_train_data['MasVnrArea'].mean()}, 
                         inplace = True)
houses_test_data.fillna({'TotalBsmtSF': 0, 'GarageCars': 2, 
                         'MasVnrArea': houses_test_data['MasVnrArea'].mean()}, inplace = True)

#Confirming, that we do not have missing values in train dataset any more.
houses_train_data.isnull().sum().sort_values(ascending = False).head()
#Confirming, that we do not have missing values in test dataset any more.
houses_test_data.isnull().sum().sort_values(ascending = False).head()
print(houses_train_data['SalePrice'].describe())
houses_train_data['SalePrice'].hist(bins = 30)
houses_train_data.columns
all_houses = pd.concat([houses_train_data, houses_test_data])
all_houses.dropna().nunique()
sns.countplot(all_houses['Street'])
all_houses[all_houses['Street'] == 'Grvl']
houses_train_data.drop(['Street'], axis = 1, inplace = True)
houses_test_data.drop(['Street'], axis = 1, inplace = True)
sns.countplot(all_houses['LandSlope'])
sns.boxplot(all_houses['LandSlope'], all_houses['SalePrice'])
houses_train_data.drop(['LandSlope'], axis = 1, inplace = True)
houses_test_data.drop(['LandSlope'], axis = 1, inplace = True)
sns.countplot(all_houses['CentralAir'])
sns.boxplot(all_houses['CentralAir'], all_houses['SalePrice'])
houses_train_data['CentralAir'].replace(['Y', 'N'], [1, 0], inplace = True)
houses_test_data['CentralAir'].replace(['Y', 'N'], [1, 0], inplace = True)
houses_train_data.columns
houses_train_data[['MSSubClass', 'LotArea', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 
                                'MasVnrArea', 'SalePrice']].corr()
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 1, 1])

axes.plot(houses_train_data['YearBuilt'], houses_train_data['SalePrice'], 'b.', alpha = 0.5)
axes.plot(houses_train_data['YearRemodAdd'], houses_train_data['SalePrice'], 'r.', alpha = 0.5)
axes.set_xlabel('YearBuilt (blue), YearRemodAdd (red)')
axes.set_ylabel('SalePrice')
houses_train_data[['TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'SalePrice']].corr()
houses_train_data[['FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 
                   'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'SalePrice']].corr()
houses_train_data[['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
       'ScreenPorch', 'PoolArea', 'MiscVal', 'SalePrice']].corr()
# List of columns, where we need to replace values
categ_columns = ['LotShape', 'LandContour', 'LotConfig',
       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
       'RoofStyle', 'RoofMatl', 'Foundation', 'Heating', 'HeatingQC', 'PavedDrive']
# Loop that replaces string values with numbers
for item in categ_columns:
    to_replace = houses_train_data[item].unique()
    values = list(range(len(pd.Series(houses_train_data[item].unique()))))
    houses_train_data[item].replace(to_replace, values, inplace = True)
    houses_test_data[item].replace(to_replace, values, inplace = True)
# Checking correlation
houses_train_data[['LotShape', 'LandContour', 'LotConfig',
       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
       'RoofStyle', 'RoofMatl', 'Foundation', 'Heating', 'HeatingQC', 'PavedDrive', 'SalePrice']].corr()
#dropping columns which weak impact on sales price
houses_train_data.drop(['EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MSSubClass',
                        'LotArea', 'YearRemodAdd', 'LowQualFinSF', '1stFlrSF', 'HalfBath', 'BedroomAbvGr',
                        'KitchenAbvGr', 'LotShape', 'LandContour', 'LotConfig', 'Neighborhood', 'Condition1', 
                        'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Heating', 
                        'PavedDrive'], axis = 1, inplace = True)
houses_test_data.drop(['EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MSSubClass',
                       'LotArea', 'YearRemodAdd', 'LowQualFinSF', '1stFlrSF', 'HalfBath', 'BedroomAbvGr',
                       'KitchenAbvGr', 'LotShape', 'LandContour', 'LotConfig', 'Neighborhood', 'Condition1', 
                       'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Heating', 
                       'PavedDrive'], axis = 1, inplace = True)
houses_train_data.columns
houses_train_data['Date'] = pd.to_datetime(houses_train_data['MoSold'].astype('str') + ' ' + houses_train_data['YrSold'].astype('str'))
houses_test_data['Date'] = pd.to_datetime(houses_test_data['MoSold'].astype('str') + ' ' + houses_test_data['YrSold'].astype('str'))
plt.figure(figsize = (12, 6))
sns.lineplot(houses_train_data['Date'], houses_train_data['SalePrice'])
# Dropping columns which month and year
houses_train_data.drop(['MoSold', 'YrSold'], axis = 1, inplace = True)
houses_test_data.drop(['MoSold', 'YrSold'], axis = 1, inplace = True)

# Saving date in different variables
dates_in_houses_train_data = houses_train_data.pop('Date')
dates_in_houses_test_data = houses_test_data.pop('Date')
# Let's check that all variables affect sales price using heatmap and correlation values
plt.figure(figsize = (10, 10))
sns.heatmap(houses_train_data.corr())
houses_train_data.corr()
# Uploading prediction libraries
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
# Test split for model validation
X = houses_train_data.drop(['SalePrice'], axis = 1)
y = houses_train_data['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40, random_state = 100)
# Creating classificators
forest_reg = RandomForestRegressor()
booster_reg = GradientBoostingRegressor()
# Setting parameters for further search
# parameters for forest_reg
parameters_forest_reg = {'n_estimators': range(50, 300, 50), 'max_depth': range(5, 30, 2)}
# parameters for booster_reg
parameters_booster_reg = {'n_estimators': range(50, 300, 50)}
# Searching for best classificator settings
search_forest_reg = GridSearchCV(forest_reg, parameters_forest_reg, cv = 5)
search_booster_reg = GridSearchCV(booster_reg, parameters_booster_reg, cv = 5)
search_forest_reg.fit(X_train,y_train)
search_booster_reg.fit(X_train,y_train)
# Checking best settings

best_forest_reg = search_forest_reg.best_estimator_
best_booster_reg = search_booster_reg.best_estimator_

print(best_forest_reg)
print(best_booster_reg)
# Let's add random_state parameter and tune model a little bit more
best_forest_reg = RandomForestRegressor(max_depth = 17, n_estimators = 100, random_state = 50)
best_booster_reg = GradientBoostingRegressor(n_estimators = 100, random_state = 50)
# Making predictions
best_forest_reg.fit(X_train, y_train)
best_booster_reg.fit(X_train, y_train)
forest_prediction = best_forest_reg.predict(X_test)
booster_prediction = best_booster_reg.predict(X_test)
# Cheching distribution of differences
(y_test - forest_prediction).hist(alpha = 0.5, bins = 30)
(y_test - booster_prediction).hist(alpha = 0.5, bins = 30)
X_train = houses_train_data.drop(['SalePrice'], axis = 1)
y_train = houses_train_data['SalePrice']
X_test = houses_test_data
best_booster_reg.fit(X_train, y_train)
booster_prediction = best_booster_reg.predict(X_test)
# Sales prices distribution in original dataset
houses_train_data['SalePrice'].describe()
# Distribution of predicted sales prices
pd.Series(booster_prediction).describe()
submission = pd.DataFrame({'Id': houses_test_data.index, 'SalePrice': booster_prediction})
submission.head()
submission.to_csv('/kaggle/working/submission.csv', index = False)