# import data

import pandas as pd



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



print("Data dimension: ",train.shape)

train.describe()
# deal with NA values

NAs = pd.concat([train.isnull().sum(), test.isnull().sum()], axis=1, keys=['Train', 'Test'])



# only deal with colums has over 5% missing values of total rows

# threshold = 0.05*train.shape[0]

print("Columns with missing values: \n", NAs[NAs.sum(axis=1) > 0])





def fillMissingValue(data):

    # see data description

    # train.Alley.unique(), [nan, 'Grvl', 'Pave']

    data['Alley'] = data['Alley'].fillna('NoAccess')

    

    #BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2, BsmtQual

    for col in ('BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual'):

        data[col] = data[col].fillna('NoBSMT')

    

    #Fence

    data['Fence'] = data['Fence'].fillna('NoFence')

    

    #FireplaceQu

    #GarageQual, GarageCond, GarageFinish, GarageType

    for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

        data[col] = data[col].fillna('NoGarage')

    #GarageYrBlt

    data['GarageYrBlt'] = data['GarageYrBlt'].fillna(0)

    

    #LotFrontage, fill with median of same zoning, 'MSZoning'

    data['LotFrontage'] = data['LotFrontage'].fillna(data['LotFrontage'].median())

    

    #MiscFeature

    data['MiscFeature'] = data['MiscFeature'].fillna('None')

    #PoolQC

    data['PoolQC'] = data['PoolQC'].fillna('NoPool')

    

    # MSZoning NA in pred. filling with most popular values

    data['MSZoning'] = data['MSZoning'].fillna(data['MSZoning'].mode()[0])

    

    # MasVnrType NA in all. filling with most popular value

    data['MasVnrType'] = data['MasVnrType'].fillna(data['MasVnrType'].mode()[0])

    

    # TotalBsmtSF  NA in pred. I suppose NA means 0

    data['TotalBsmtSF'] = data['TotalBsmtSF'].fillna(0)



    # Electrical NA in pred. filling with most popular values

    data['Electrical'] = data['Electrical'].fillna(data['Electrical'].mode()[0])

    

    # KitchenQual NA in pred. filling with most popular values

    data['KitchenQual'] = data['KitchenQual'].fillna(data['KitchenQual'].mode()[0])



    # FireplaceQu  NA in all. NA means No Fireplace

    data['FireplaceQu'] = data['FireplaceQu'].fillna('NoFireplace')

    

    # GarageCars  NA in pred. I suppose NA means 0

    data['GarageCars'] = data['GarageCars'].fillna(0.0)



    # SaleType NA in pred. filling with most popular values

    data['SaleType'] = data['SaleType'].fillna(data['SaleType'].mode()[0])

    

    print(type(data))

    return data

 



train = fillMissingValue(train)

test = fillMissingValue(test)



# save to csv

train.to_csv('../working/train_fillna.csv', sep=',', index=False)

test.to_csv('../working/test_fillna.csv', sep=',', index=False)



#fillNA(train)

#fillNA(test)

# fill missing values
# look at numeric columns first



train_numeric = train._get_numeric_data()

train_category = train[train.columns.difference(train_numeric.columns)]

print(type(train_category))

train_numeric.drop('Id', axis=1, inplace=True)





test_numeric = test._get_numeric_data()

test_category = test[test.columns.difference(test_numeric.columns)]

print(type(test_category))

test_numeric.drop('Id', axis=1, inplace=True)

test_numeric.shape



train_numeric.to_csv('../working/train_numeric.csv', sep=',', index=False)

test_numeric.to_csv('../working/test_numeric.csv', sep=',', index=False)



train_category.to_csv('../working/train_category.csv', sep=',', index=False)

test_category.to_csv('../working/test_category.csv', sep=',', index=False)



# save category as well
# simple regression

import numpy as np

from sklearn.linear_model import LinearRegression



data = np.asarray(train_numeric)

lr = LinearRegression()

X, y = data[:, 0:35], data[:, 36]



lr.fit(X, y)

LinearRegression(copy_X=True, fit_intercept=True, normalize=False)

lr.coef_

lr.intercept_