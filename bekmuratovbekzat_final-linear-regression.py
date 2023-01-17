# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_ds = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test_ds = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

train_ds.head()
#scatter plot grlivarea/saleprice

var = 'GarageArea'

data = pd.concat([train_ds['SalePrice'], train_ds['GarageArea']], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
sns.set()

cols = ['SalePrice','OverallQual',

                        '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea']

sns.pairplot(train_ds[cols], height = 2.5)

plt.show();
sns.set()

cols = ['SalePrice', 'FullBath', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea', 

                         'TotalBsmtSF', 'YearBuilt']

sns.pairplot(train_ds[cols], height = 2.5)

plt.show();


corrmat = train_ds.corr()



#saleprice correlation matrix

k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train_ds[cols].values.T)

sns.set(font_scale=1.)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
edited_train = train_ds[['SalePrice', 'SaleCondition', 'OverallCond',

                        'OverallQual',

                        '1stFlrSF', '2ndFlrSF', 'GrLivArea',

                        'FullBath', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea', 

                         'TotalBsmtSF', 'YearBuilt', 'KitchenQual', 'GarageQual', 'Condition1', 'Condition2']]

edited_train.head()
edited_train = pd.get_dummies(edited_train, columns=['OverallQual', 'SaleCondition', 'OverallCond', 'KitchenQual', 'GarageQual',

                                                    'Condition1', 'Condition2'])

# edited_train = edited_train.drop(columns = ["Condition2_RRAe", "Condition2_RRAn", "Condition2_RRNn", "HouseStyle_2.5Fin", 

#                                             'Utilities_NoSeWa', 'Electrical_Mix', 'Electrical_Mix', 'GarageQual_Ex', 'Utilities_NoSeWa',

#                                            'Heating_OthW', 'Heating_Floor', 'PoolQC_Fa'], axis = 1)

edited_train = edited_train.drop(columns = ["Condition2_RRAe", "Condition2_RRAn", "Condition2_RRNn",'GarageQual_Ex',], axis = 1)

edited_train.head()
x = edited_train.iloc[:, 1:].values

y = edited_train.iloc[:, 0].values
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix

from sklearn.linear_model import LinearRegression
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 10)
regressor = LinearRegression()

regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
from sklearn.metrics import mean_squared_error, r2_score
print('mean_squared_error: ', mean_squared_error(y_test, y_pred),

     '\nr2_score: ',r2_score(y_test, y_pred)

     )
test_ds.head()
test = test_ds[['SaleCondition', 'LotArea', 'LandContour', 'LotFrontage','LotShape', 'MSSubClass', 'YrSold', 'SaleType', 

                         'MSZoning', 'Street', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual',

                        'OverallCond', 'ExterQual', 'ExterCond', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',

                        'FullBath', 'HalfBath', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'GarageType',

                        'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive',

                        'MoSold', 'YearBuilt', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch','ScreenPorch',

                        'PoolArea', 'Foundation', 'LandSlope', 'Utilities', 'LotConfig', 'BsmtQual', 'BsmtCond', 

                         'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Fence', 'PoolQC']]

test.LotFrontage = test.LotFrontage.fillna(value = test.LotFrontage.mean())

test.Functional = test.Functional.fillna(value = 'Typ')

test.GarageType = test.GarageType.fillna(value = 'Basment')

test.GarageFinish = test.GarageFinish.fillna(value = 'Fin')

test.GarageQual = test.GarageQual.fillna(value = 'TA')

test.GarageCond = test.GarageCond.fillna(value = 'TA')

test.GarageArea = test.GarageArea.fillna(value = 0.0)

test.GarageCars = test.GarageCars.fillna(value = 0)

test.BsmtQual = test.BsmtQual.fillna(value = 'TA')

test.BsmtCond = test.BsmtCond.fillna(value = 'TA')

test.TotalBsmtSF = test.TotalBsmtSF.fillna(value = test.TotalBsmtSF.mean())

test.Utilities = test.Utilities.fillna(value = 'AllPub')



test.isnull().sum()

test = pd.get_dummies(test, columns=['SaleType','LotShape', 'LandContour','SaleCondition', 'MSZoning', 'Street', 

                                                     'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual',

                                                    'OverallCond', 'ExterQual', 'ExterCond', 'Electrical', 'KitchenQual', 

                                                    'Functional', 'GarageType', 'GarageFinish', 'GarageCond',

                                                    'GarageQual', 'PavedDrive', 'Foundation',

                                                    'LandSlope', 'Utilities', 'LotConfig', 'BsmtQual', 'BsmtCond', 

                                                    'Heating', 'HeatingQC', 'CentralAir', 'Fence', 'PoolQC'])

test.shape
test = test_ds[['OverallQual',

                        '1stFlrSF','2ndFlrSF', 'GrLivArea',

                        'FullBath', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea', 

                         'TotalBsmtSF', 'YearBuilt', 'SaleCondition', 'OverallCond', 'KitchenQual', 'GarageQual',

                                                    'Condition1', 'Condition2']]



test.isnull().sum()

test.TotalBsmtSF = test.TotalBsmtSF.fillna(value = 0.0)

test.GarageArea = test.GarageArea.fillna(value = 0.0)

test.GarageCars = test.GarageCars.fillna(value = 0)

test = pd.get_dummies(test, columns=['OverallQual', 'SaleCondition', 'OverallCond', 'KitchenQual', 'GarageQual', 'Condition1', 'Condition2'])

test.shape
x.shape
y_pred = regressor.predict(test)
# print('mean_squared_error: ', mean_squared_error(y_test, y_pred),

#      '\nr2_score: ',r2_score(y_test, y_pred)

#      )
sub = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')

sub['SalePrice'] = y_pred

sub.to_csv('mde.csv', index=False)