#!/usr/bin/python3

# coding=utf-8

# import warnings filter

from warnings import simplefilter

# ignore all future warnings

simplefilter(action='ignore', category=FutureWarning)
#!/usr/bin/python3

# coding=utf-8

#===========================================================================

# load up the libraries

#===========================================================================

import pandas  as pd



#===========================================================================

# read in the data

#===========================================================================

train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test_data  = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')



#===========================================================================

# select some features to rank. These are all 'integer' fields for today.

#===========================================================================

features = ['MSSubClass', 'LotArea', 'OverallQual', 'OverallCond', 

        'YearBuilt', 'YearRemodAdd', 'BsmtFinSF1', 'BsmtFinSF2', 

        'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 

        'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 

        'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 

        'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 

        'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 

        'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']



#===========================================================================

#===========================================================================

X_train       = train_data[features]

y_train       = train_data["SalePrice"]

final_X_test  = test_data[features]



#===========================================================================

# simple preprocessing: imputation; substitute any 'NaN' with mean value

#===========================================================================

X_train      = X_train.fillna(X_train.mean())

final_X_test = final_X_test.fillna(final_X_test.mean())



#===========================================================================

# set up our regressor + fit. 

# Today we shall be using the random forest regressor

#===========================================================================

from sklearn.ensemble import RandomForestRegressor



regressor = RandomForestRegressor(n_estimators=100, max_depth=10)

regressor.fit(X_train, y_train)



#===========================================================================

# perform the PermutationImportance

#===========================================================================

import eli5

from eli5.sklearn import PermutationImportance



perm_import = PermutationImportance(regressor, random_state=1).fit(X_train, y_train)



# visualize the results

eli5.show_weights(perm_import, top=None, feature_names = X_train.columns.tolist())
#===========================================================================

# perform a scikit-learn Recursive Feature Elimination (RFE)

#===========================================================================

from sklearn.feature_selection import RFE

# here we want only one final feature, we do this to produce a ranking

rfe = RFE(regressor, n_features_to_select=1)

rfe.fit(X_train, y_train)



#===========================================================================

# now print out the features in order of ranking

#===========================================================================

from operator import itemgetter

for x, y in (sorted(zip(rfe.ranking_ , features), key=itemgetter(0))):

    print(x, y)