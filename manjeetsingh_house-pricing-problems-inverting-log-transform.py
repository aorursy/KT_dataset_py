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
# Import the general libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
# Import the machine learning libraries

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_log_error, mean_squared_error
# Import the datasets

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
# Combine the datasets into one dataframe for preprocessing

# (So they will have the same shape after making dummy variables)

df = pd.concat((train, test), ignore_index=True)
# Define function to impute missing data:

def impute_missing_data(df):



    # drop the 'MiscFeature' column inplace

    df.drop('MiscFeature', axis=1, inplace=True)



    # handle the simple missing values

    df.loc[df.MasVnrType.isnull(), 'MasVnrType'] = 'None' # no good

    df.loc[df.MasVnrType == 'None', 'MasVnrArea'] = 0

    df.loc[df.LotArea.isnull(), 'MasVnrType'] = 0

    df.loc[df.BsmtQual.isnull(), 'BsmtQual'] = 'NoBsmt'

    df.loc[df.BsmtCond.isnull(), 'BsmtCond'] = 'NoBsmt'

    df.loc[df.BsmtExposure.isnull(), 'BsmtExposure'] = 'NoBsmt'

    df.loc[df.BsmtFinType1.isnull(), 'BsmtFinType1'] = 'NoBsmt'

    df.loc[df.BsmtFinType2.isnull(), 'BsmtFinType2'] = 'NoBsmt'

    df.loc[df.BsmtFinType1=='NoBsmt', 'BsmtFinSF1'] = 0

    df.loc[df.BsmtFinType2=='NoBsmt', 'BsmtFinSF2'] = 0

    df.loc[df.BsmtFinSF1.isnull(), 'BsmtFinSF1'] = df.BsmtFinSF1.median()

    df.loc[df.BsmtQual=='NoBsmt', 'BsmtUnfSF'] = 0

    df.loc[df.BsmtUnfSF.isnull(), 'BsmtUnfSF'] = df.BsmtUnfSF.median()

    df.loc[df.BsmtQual=='NoBsmt', 'TotalBsmtSF'] = 0

    df.loc[df.FireplaceQu.isnull(), 'FireplaceQu'] = 'NoFireplace'

    df.loc[df.GarageType.isnull(), 'GarageType'] = 'NoGarage'

    df.loc[df.GarageFinish.isnull(), 'GarageFinish'] = 'NoGarage'

    df.loc[df.GarageQual.isnull(), 'GarageQual'] = 'NoGarage'

    df.loc[df.GarageCond.isnull(), 'GarageCond'] = 'NoGarage'

    df.loc[df.BsmtFullBath.isnull(), 'BsmtFullBath'] = 0

    df.loc[df.BsmtHalfBath.isnull(), 'BsmtHalfBath'] = 0

    df.loc[df.KitchenQual.isnull(), 'KitchenQual'] = 'TA'

    df.loc[df.MSZoning.isnull(), 'MSZoning'] = 'RL'

    df.loc[df.Utilities.isnull(), 'Utilities'] = 'AllPub'

    df.loc[df.Exterior1st.isnull(), 'Exterior1st'] = 'VinylSd'

    df.loc[df.Exterior2nd.isnull(), 'Exterior2nd'] = 'VinylSd'

    df.loc[df.Functional.isnull(), 'Functional'] = 'Typ'

    df.loc[df.SaleCondition.isnull(), 'SaleCondition'] = 'Normal'

    df.loc[df.SaleCondition.isnull(), 'SaleType'] = 'WD'

    df.loc[df['Electrical'].isnull(), 'Electrical'] = 'SBrkr'

    df.loc[df['SaleType'].isnull(), 'SaleType'] = 'NoSale'

    df.loc[df['Alley'].isnull(), 'Alley'] = 'NA'

    df.loc[df['PoolQC'].isnull(), 'PoolQC'] = 'NA'

    df.loc[df['Fence'].isnull(), 'Fence'] = 'NA'



    # Ken - GarageYrBlt --> Make same as year built for homes with no garage

    df.loc[df.GarageYrBlt.isnull(), 'GarageYrBlt'] = df['YearBuilt']



    # only one is null and it has type Detchd

    df.loc[df['GarageArea'].isnull(), 'GarageArea'] = df.loc[df['GarageType']=='Detchd', 'GarageArea'].mean()

    df.loc[df['GarageCars'].isnull(), 'GarageCars'] = df.loc[df['GarageType']=='Detchd', 'GarageCars'].median()



    # LotFrontage:

    # Generate a dictionary where the types of Lot Configurations are the keys and the ratio of total LotArea to

    # LotFrontage are the values

    frontageratios = dict((df.groupby('LotConfig')['LotArea'].mean() / df.groupby('LotConfig')['LotFrontage'].mean()).astype(int))



    # Impute missing LotFrontage values by applying applicable frontageratio to the lot's area

    df.loc[df.LotFrontage.isnull(), 'LotFrontage'] = (df['LotArea'] / df['LotConfig'].map(frontageratios)).astype(int)



    # CentralAir

    size_mapping = {'Y': 1,'N': 0}

    df['CentralAir'] = df['CentralAir'].map(size_mapping)



    return df
# Define function to change categorical variables to dummy variables

def make_dummies(df):



    # Encode the Categorical Data

    # Encode the Feature Variables (don't need to code the target)

    catcols = ['MSSubClass','MSZoning','Street','Alley','LotShape',

               'LandContour','Utilities','LotConfig','LandSlope',

               'Neighborhood','Condition1','Condition2','BldgType',

               'HouseStyle','RoofStyle','RoofMatl','Exterior1st',

               'Exterior2nd','MasVnrType','ExterQual','ExterCond',

               'Foundation','BsmtQual','BsmtCond','BsmtExposure',

               'BsmtFinType1','BsmtFinType2','Heating','HeatingQC',

               'CentralAir','Electrical','KitchenQual','Functional',

               'FireplaceQu','GarageType','GarageFinish','GarageQual',

               'GarageCond','PavedDrive','PoolQC','Fence',

               'SaleType','SaleCondition']



    return pd.get_dummies(df, columns=catcols, drop_first=True)
# Use custom function (above) to impute the missing values

df = impute_missing_data(df)
# Use custom function (above) to change categorical features to dummy variables

df = make_dummies(df)
# Split back into separate datasets

train = df[:train.shape[0]]

test = df[train.shape[0]:]



# Drop the 'SalePrice' column from the test dataset (it was added during combination)

test = test.drop('SalePrice', axis=1)
X = train.drop('SalePrice', axis=1)

y = train['SalePrice']
# Logtransform y variable

y = np.log1p(y)
# Use train-test-split to create 'local' training and testing datasets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
# Create and fit the regressor object

model = LinearRegression()

model.fit(X_train, y_train)
# Predict the test set results

y_pred = model.predict(X_test)



# Eliminate negative results by setting them to zero

y_pred[y_pred<0] = 0
# Looking at the scores of the mdoel on the test set

r2 = model.score(X_test, y_test)

RMSLE = np.sqrt(mean_squared_log_error(y_test, y_pred))

RMSE = np.sqrt(mean_squared_error(y_test, y_pred))



# Show results from local train-test split

print()

print("Results from local train-test split:")

print("r^2: {}.".format(r2))

print("RMSLE: {}.".format(RMSLE))

print("RMSE: {}.".format(RMSE))

print()
# Run the model's prediction method on the test dataset

predictions = model.predict(test)

predictions[predictions<0] = 0
predictions = np.expm1(predictions)
predictions.min()
predictions.max()
list(predictions).count('inf')
list(predictions).count(-1)
len(predictions)
submission = test[['Id']].copy()
submission['SalePrice'] = predictions
submission[submission.isin([np.inf, -np.inf, -1]).any(1)]
submission.to_csv('submission_test2.csv', index=False, header=True)