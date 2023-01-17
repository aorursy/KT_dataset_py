import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
%matplotlib inline
#get training and testing data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
#Clean Training Data
for col in ('GarageYrBlt','GarageArea', 'GarageCars'):
    train[col] = train[col].fillna(0)

for col in ('BsmtFinSF1','BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF','BsmtFullBath',
            'BsmtHalfBath'):
    train[col] = train[col].fillna(0)
    
train["MasVnrArea"] = train["MasVnrArea"].fillna(0)

for col in ('BsmtQual', 'BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2'):
    train[col] = train[col].fillna('None')

train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']
train ['MSZoning'] = train['MSZoning'].fillna(train['MSZoning'].mode()[0])
train["Functional"] = train["Functional"].fillna('Typ')
train['Electrical'] = train['Electrical'].fillna(train['Electrical'].mode()[0])
train['KitchenQual'] =train['KitchenQual'].fillna(train['KitchenQual'].mode()[0])
train ['Exterior1st']= train['Exterior1st'].fillna(train['Exterior1st'].mode()[0])
train['Exterior2nd']= train['Exterior2nd'].fillna(train['Exterior2nd'].mode()[0])
train['SaleType'] = train['SaleType'].fillna(train['SaleType'].mode()[0])
train = train.drop(['Utilities'], axis=1)
#Clean Test Data
for col in ('GarageYrBlt','GarageArea', 'GarageCars'):
    test[col] = test[col].fillna(0)

for col in ('BsmtFinSF1','BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF','BsmtFullBath',
            'BsmtHalfBath'):
    test[col] = test[col].fillna(0)
    
test["MasVnrArea"] = test["MasVnrArea"].fillna(0)

for col in ('BsmtQual', 'BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2'):
    test[col] = test[col].fillna('None')

test['TotalSF'] = test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF']
test ['MSZoning'] = test['MSZoning'].fillna(test['MSZoning'].mode()[0])
test ['LotFrontage'] = test['LotFrontage'].fillna(test['LotFrontage'].mode()[0])
test["Functional"] = test["Functional"].fillna('Typ')
test['Electrical'] = test['Electrical'].fillna(test['Electrical'].mode()[0])
test['KitchenQual'] =test['KitchenQual'].fillna(test['KitchenQual'].mode()[0])
test ['Exterior1st']= test['Exterior1st'].fillna(test['Exterior1st'].mode()[0])
test['Exterior2nd']= test['Exterior2nd'].fillna(test['Exterior2nd'].mode()[0])
test['SaleType'] = test['SaleType'].fillna(test['SaleType'].mode()[0])
test = test.drop(['Utilities'], axis=1)
#Choose Features to evaluate in Model, this case only picked numerical features
numeric_features_test = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',
       'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
       'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
       'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
       'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
       'MoSold', 'YrSold', 'TotalSF']
#assign training and testing values to variables
train_X = train[numeric_features_test]
train_y = train['SalePrice']
val_X = test[numeric_features_test]
#create model
train_model = RandomForestRegressor(random_state=1)
#fit model with training values
train_model.fit(train_X,train_y)
#get housing predictions based on test features chosen
housing_preds = train_model.predict(val_X)
#Format Output and save in submission.csv
output = pd.DataFrame({'Id': test.Id, 'SalePrice': housing_preds})
output.to_csv('submission.csv', index=False)


