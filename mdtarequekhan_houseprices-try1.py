# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

from sklearn.linear_model import LogisticRegression , LinearRegression
# attributes to retain, found manually by looking at visualization from weka

cols = ['Id', 'MSSubClass', 'LotArea', 'LotShape', 'LotConfig', 'Neighborhood', 'HouseStyle', 

        'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'Exterior1st',

        'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 

        'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtUnfSF', 'TotalBsmtSF',

        'HeatingQC', 'Electrical', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'BedroomAbvGr', 

        'KitchenQual', 'TotRmsAbvGrd', 'Fireplaces', 'FireplaceQ', 'GarageType', 'GarageFinish',

        'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'Fence', 'MoSold', 'YrSold', 

        'SaleCondition' , 'SalePrice']



data = pd.read_csv('../input/train.csv', dtype = {'SalePrice': float})

test = pd.read_csv('../input/test.csv')
test.head()
# 1. drop manually found useless columns via weka

# 2. drop non-numeric columns from remaining set of attributes



# Get out X and y first

X = data.drop('SalePrice', axis=1)

y = data['SalePrice']
non_numeric_attributes = []

for c in X.columns:

    if c == 'Id' or c == 'SalePrice': continue

    if str(X[c].dtype) != 'int64':

        # print c, '--', X[c].dtype, ' : dropping'

        non_numeric_attributes.append(c)

        

X.drop(non_numeric_attributes, axis=1, inplace=True)



print('Dropped ',len(non_numeric_attributes),' non numeric attributes')

print( len(X.columns), ' numeric attributes remaining' )
attributes_to_drop = [ x for x in data.columns if x not in cols]

print (attributes_to_drop)

X.drop(attributes_to_drop, axis=1, inplace=True, errors='ignore')

X.head()
print (len(X.columns), ' numeric attributes remaining')

print (X.shape)
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=3)
print( X_train.shape )

print( y_train.shape )

print( X_test.shape )

print( y_test.shape )
logreg = LogisticRegression()

logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
from sklearn import metrics

import numpy as np

print ( 'RMSE ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)) )
from sklearn.cross_validation import cross_val_score

lr = LogisticRegression()

print(   np.sqrt(-cross_val_score(lr, X,y, cv=5, scoring='mean_squared_error' ) ).mean() )