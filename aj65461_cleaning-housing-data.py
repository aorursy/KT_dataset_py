import numpy as np 

import pandas as pd 

import warnings

import os

import scipy.stats as stats

warnings.filterwarnings('ignore')
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.info()
test.info()
numVar = ['YearRemodAdd', 'OverallQual',

      'GrLivArea', 'GarageCars',

      'GarageArea', 'TotalBsmtSF',

      '1stFlrSF', 'FullBath',

      'TotRmsAbvGrd', 'YearBuilt']
train[numVar].isnull().sum(axis = 0)
test[numVar].isnull().sum(axis = 0)
test[numVar][test['GarageCars'].isnull()]
test.iloc[1116]
test['GarageCars'] = test['GarageCars'].fillna(0)

test['GarageArea'] = test['GarageArea'].fillna(0)
test[numVar][test['TotalBsmtSF'].isnull()]
test.loc[[660],['TotalBsmtSF','BsmtQual','BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF']]
test['TotalBsmtSF'] = test['TotalBsmtSF'].fillna(0)
catVar = train.select_dtypes(include = [np.object]).columns
train[catVar].isnull().sum(axis = 0)
test[catVar].isnull().sum(axis = 0)
test[catVar][test['MSZoning'].isnull()]
test['MSZoning'].mode()[0]
# Store the most frequent value for MSZoning for readability

zoning_mode = str(test['MSZoning'].mode()[0])



test['MSZoning'] = test['MSZoning'].fillna(zoning_mode)
train[catVar]['Alley'].unique()
train['Alley'] = train['Alley'].fillna('None')

test['Alley'] = test['Alley'].fillna('None')
test[catVar][test['Utilities'].isnull()]
test.loc[[455,485],['Electrical']]
utilities_mode = str(test['Utilities'].mode()[0])

utilities_mode
test['Utilities'] = test['Utilities'].fillna(utilities_mode)
test[catVar][test['Exterior1st'].isnull()]
test.loc[[691],['Exterior1st', 'Exterior2nd']]
ext1_mode = str(test['Exterior1st'].mode()[0])

ext1_mode
ext2_mode = str(test['Exterior2nd'].mode()[0])

ext2_mode
criteria = test['Exterior1st'] == 'VinylSd'

vinyl = test[criteria]
vinyl.loc[:, ['Exterior1st', 'Exterior2nd']].head()
criteria2 = test['Neighborhood'] == 'Edwards'

edwards = test[criteria2]
edwards['Exterior1st'].value_counts()
edwards['Exterior2nd'].value_counts()
ext_edwards_mode = str(edwards['Exterior1st'].mode()[0])

ext_edwards_mode
test['Exterior1st'] = test['Exterior1st'].fillna(ext_edwards_mode)

test['Exterior2nd'] = test['Exterior2nd'].fillna(ext_edwards_mode)
train[catVar]['MasVnrType'].unique()
train[catVar][train['MasVnrType'].isnull()].loc[:, ['MasVnrType', 'MasVnrArea']]
train['MasVnrType'] = train['MasVnrType'].fillna('None')
test[catVar][test['MasVnrType'].isnull()].loc[:, ['MasVnrType', 'MasVnrArea']]
test['MasVnrType'] = test['MasVnrType'].fillna('None')
# Store basement variables for easy reference

basement_variables = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']



train[train[basement_variables[0]].isnull()].loc[:, basement_variables]
train[basement_variables] = train[basement_variables].fillna('None')
test[test[basement_variables[0]].isnull()].loc[:, basement_variables]
bsmtqual_mode = str(test['BsmtQual'].mode()[0])

bsmtqual_mode
test.at[757, 'BsmtQual'] = bsmtqual_mode

test.at[758, 'BsmtQual'] = bsmtqual_mode
test[basement_variables] = test[basement_variables].fillna('None')
train[catVar][train['Electrical'].isnull()].loc[:, ['Utilities', 'Electrical']]
criteria3 = train['Utilities'] == 'AllPub'

all_pub = train[criteria3]
all_pub['Electrical'].describe()
allpub_mode = str(all_pub['Electrical'].mode()[0])

allpub_mode
train['Electrical'] = train['Electrical'].fillna(allpub_mode)
test[catVar][test['KitchenQual'].isnull()].loc[:, ['KitchenAbvGr', 'KitchenQual']]
test['KitchenAbvGr'].value_counts()
kitqual_mode = str(test['KitchenQual'].mode()[0])

kitqual_mode
test['KitchenQual'] = test['KitchenQual'].fillna(kitqual_mode)
test[catVar][test['Functional'].isnull()]
test[test['Neighborhood'] == 'IDOTRR']['Functional']
functional_mode = str(test['Functional'].mode()[0])

functional_mode
test['Functional'] = test['Functional'].fillna(functional_mode)
train[catVar][train['FireplaceQu'].isnull()].loc[:, ['Fireplaces', 'FireplaceQu']]['Fireplaces'].unique()
train['FireplaceQu'] = train['FireplaceQu'].fillna('None')
test[catVar][test['FireplaceQu'].isnull()].loc[:, ['Fireplaces', 'FireplaceQu']]['Fireplaces'].unique()
test['FireplaceQu'] = test['FireplaceQu'].fillna('None')
# Store garage variables for easy reference

garage_variables = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']



len(train[train[garage_variables[0]].isnull()].loc[:, garage_variables])
train[garage_variables] = train[garage_variables].fillna('None')
test[test[garage_variables[0]].isnull()].loc[:, garage_variables]
test[garage_variables] = test[garage_variables].fillna('None')
train[train['PoolQC'].isnull()].loc[:, 'PoolArea']
train['PoolQC'] = train['PoolQC'].fillna('None')

test['PoolQC'] = test['PoolQC'].fillna('None')
train['Fence'] = train['Fence'].fillna('None')

test['Fence'] = test['Fence'].fillna('None')
train['MiscFeature'] = train['MiscFeature'].fillna('None')

test['MiscFeature'] = test['MiscFeature'].fillna('None')
saleType_mode = str(test['SaleType'].mode()[0])

saleType_mode
test['SaleType'] = test['SaleType'].fillna(saleType_mode)
train[catVar].isnull().sum(axis = 0)
test[catVar].isnull().sum(axis = 0)
class ChiSquare:

    def __init__(self, dataframe):

        self.df = dataframe

        self.p = None #P-Value

        self.chi2 = None #Chi Test Statistic

        self.dof = None

        

        self.dfObserved = None

        self.dfExpected = None

        

    def _print_chisquare_result(self, colX, alpha):

        result = ""

        if self.p<alpha:

            result="{0} is IMPORTANT for Prediction".format(colX)

        else:

            result="{0} is NOT an important predictor. (Discard {0} from model)".format(colX)



        print(result)

        

    def TestIndependence(self,colX,colY, alpha=0.05):

        X = self.df[colX].astype(str)

        Y = self.df[colY].astype(str)

        

        self.dfObserved = pd.crosstab(Y,X) 

        chi2, p, dof, expected = stats.chi2_contingency(self.dfObserved.values)

        self.p = p

        self.chi2 = chi2

        self.dof = dof 

        

        self.dfExpected = pd.DataFrame(expected, columns=self.dfObserved.columns, index = self.dfObserved.index)

        

        self._print_chisquare_result(colX,alpha)
cT = ChiSquare(train)
for var in catVar:

    cT.TestIndependence(colX = var, colY = "SalePrice")
final_catVar = ['MSZoning', 'Street', 'LotShape', 'LotConfig', 'Neighborhood',

               'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',

                'BsmtCond', 'Heating', 'CentralAir', 'KitchenQual', 'FireplaceQu',

               'GarageFinish', 'SaleType', 'SaleCondition']
final_cols = ['Id'] + numVar + final_catVar
len(final_cols)
fin_train_cols = final_cols + ['SalePrice']

final_train = train[fin_train_cols]
final_test = test[final_cols]
final_train.info()
final_test.info()