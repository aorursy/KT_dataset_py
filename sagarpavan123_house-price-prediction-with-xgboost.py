# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

from scipy.stats import skew

import seaborn as sns

import matplotlib.pyplot as plt



from scipy.special import boxcox1p

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from xgboost import XGBRegressor



pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)
# Read the CSV files

train_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')



print('Train data shape: {}'.format(train_df.shape))

print('Test data shape: {}'.format(test_df.shape))
house_prise_df = pd.concat([train_df.drop(['Id', 'SalePrice'], axis=1), test_df.drop('Id', axis=1)], axis=0) 

house_prise_df.info()
pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)

# Checking the percentage of missing values from the data

round(100*(house_prise_df.isnull().sum()/len(house_prise_df.index)), 2)
## Imputing missing values

house_prise_df['Alley'] = house_prise_df['Alley'].replace(np.nan, 'NoAlleyAccess')

house_prise_df['PoolQC'] = house_prise_df['PoolQC'].replace(np.nan, 'NoPool')

house_prise_df['Fence'] = house_prise_df['Fence'].replace(np.nan, 'NoFence')

house_prise_df['MiscFeature'] = house_prise_df['MiscFeature'].replace(np.nan, 'NoFeature')

house_prise_df['GarageCond'] = house_prise_df['GarageCond'].replace(np.nan, 'NoGarageCond')

house_prise_df['GarageQual'] = house_prise_df['GarageQual'].replace(np.nan, 'NoGarageQual')

house_prise_df['GarageFinish'] = house_prise_df['GarageFinish'].replace(np.nan, 'NoGarageFinish')

house_prise_df['GarageType'] = house_prise_df['GarageType'].replace(np.nan, 'NoGarageType')

house_prise_df['MasVnrType'] = house_prise_df['MasVnrType'].replace(np.nan, 'NoMasVnrType')

house_prise_df['BsmtQual'] = house_prise_df['BsmtQual'].replace(np.nan, 'NoBsmtQual')

house_prise_df['BsmtCond'] = house_prise_df['BsmtCond'].replace(np.nan, 'NoBsmtCond')

house_prise_df['BsmtExposure'] = house_prise_df['BsmtExposure'].replace(np.nan, 'NoBsmtExposure')

house_prise_df['BsmtFinType1'] = house_prise_df['BsmtFinType1'].replace(np.nan, 'NoBsmtFinType1')

house_prise_df['BsmtFinType2'] = house_prise_df['BsmtFinType2'].replace(np.nan, 'NoBsmtFinType2')

house_prise_df['FireplaceQu'] = house_prise_df['FireplaceQu'].replace(np.nan, 'NoFireplaceQu')

house_prise_df['Electrical'] = house_prise_df['Electrical'].replace(np.nan, 'NoElectrical')



# Replace the missing values in each of the columns below with their mode

house_prise_df['MSZoning'] = house_prise_df.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

house_prise_df['Exterior1st'] = house_prise_df['Exterior1st'].transform(lambda x: x.fillna(x.mode()[0]))

house_prise_df['Exterior2nd'] = house_prise_df['Exterior2nd'].transform(lambda x: x.fillna(x.mode()[0]))

house_prise_df['KitchenQual'] = house_prise_df['KitchenQual'].transform(lambda x: x.fillna(x.mode()[0]))

house_prise_df["Functional"] = house_prise_df["Functional"].replace(np.nan, 'Typ')

house_prise_df['SaleType'] = house_prise_df['SaleType'].transform(lambda x: x.fillna(x.mode()[0]))



# Replace the missing values with median

house_prise_df["LotFrontage"] = house_prise_df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))



# Replace the missing values with 0

house_prise_df['MasVnrArea'] = house_prise_df['MasVnrArea'].fillna(0)

house_prise_df['BsmtFinSF1'] = house_prise_df['BsmtFinSF1'].fillna(0)

house_prise_df['BsmtFinSF2'] = house_prise_df['BsmtFinSF2'].fillna(0)

house_prise_df['BsmtUnfSF'] = house_prise_df['BsmtUnfSF'].fillna(0)

house_prise_df['TotalBsmtSF'] = house_prise_df['TotalBsmtSF'].fillna(0)

house_prise_df['BsmtFullBath'] = house_prise_df['BsmtFullBath'].fillna(0)

house_prise_df['BsmtHalfBath'] = house_prise_df['BsmtHalfBath'].fillna(0)

# Replacing the missing values with 0, since GarageType = No Garage

house_prise_df['GarageYrBlt'] = house_prise_df['GarageYrBlt'].fillna(0)

house_prise_df['GarageCars'] = house_prise_df['GarageCars'].fillna(0)

house_prise_df['GarageArea'] = house_prise_df['GarageArea'].fillna(0)



# Drop the 'Utilities' column as it contains only 2 classes with highly imbalance

house_prise_df = house_prise_df.drop(['Utilities', 'YrSold', 'MoSold'], axis=1)
# Checking the percentage of missing values from the data

round(100*(house_prise_df.isnull().sum()/len(house_prise_df.index)), 2)
house_prise_df['MSSubClass'] = house_prise_df['MSSubClass'].apply(str)



quantitative_columns = [f for f in house_prise_df.columns if house_prise_df.dtypes[f] != 'object']

qualitative_columns = [f for f in house_prise_df.columns if house_prise_df.dtypes[f] == 'object']



house_prise_df[quantitative_columns].head(100)
house_prise_df['TotalHomeQuality'] = house_prise_df['OverallQual'] + house_prise_df['OverallCond']

house_prise_df['YrBltAndRemod'] = np.where(house_prise_df['YearBuilt']>=house_prise_df['YearRemodAdd'], house_prise_df['YearBuilt'], house_prise_df['YearRemodAdd'])

house_prise_df = house_prise_df.drop(['OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd'], axis=1)
quantitative_columns = [f for f in house_prise_df.columns if house_prise_df.dtypes[f] != 'object']

house_prise_df[quantitative_columns].head(20)
# Check the skewness for numerical columns

skewed_features = house_prise_df[quantitative_columns].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

skewness = pd.DataFrame({'Skew' :skewed_features})

skewness
skewness = skewness[abs(skewness) > 0.75]

skewness.dropna(inplace=True)

skewness = skewness.drop('GarageYrBlt')

skewness
skewed_features = skewness.index

lam = 0.15

for col in skewed_features:

    house_prise_df[col] = boxcox1p(house_prise_df[col], lam)

house_prise_df = pd.get_dummies(house_prise_df)

house_prise_df.shape
# Train The Model

train_data = house_prise_df[:train_df.shape[0]]

test_data = house_prise_df[train_df.shape[0]:]



# Train test split

X_train, X_val, y_train, y_val = train_test_split(train_data, train_df.SalePrice, test_size=0.20, random_state=20)
# XGBoost Regressor

model = XGBRegressor(learning_rate=0.01,

                       n_estimators=1000,

                       max_depth=4,

                       min_child_weight=0,

                       gamma=0.6,

                       subsample=0.7,

                       colsample_bytree=0.7,

                       objective='reg:linear',

                       nthread=-1,

                       scale_pos_weight=1,

                       seed=27,

                       reg_alpha=0.00006,

                       random_state=20)
n_folds = 5



y_train_scaled = np.log1p(y_train)

y_val_scaled = np.log1p(y_val)



def rmse_cv(model, X, y):

    kf = KFold(n_folds, shuffle=True, random_state=20).get_n_splits(X.values)

    rmse = np.sqrt(-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5))

    return rmse



X = X_train

y = y_train_scaled



score = rmse_cv(model, X, y)

print("Model score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))
def rmse_cv_val(model, X, y):

    rmse = np.sqrt(-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5))

    return rmse



X = X_val

y = y_val_scaled





score = rmse_cv_val(model, X, y)

print("Validation score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# Predict on test dataset and submit



X = X_train

y = y_train_scaled

model.fit(X, y)



test_predictions = np.exp( model.predict(test_data))

test_predictions
final_result=pd.DataFrame({'Id':test_df.Id, 'SalePrice':test_predictions})

final_result.to_csv('/kaggle/working/submission.csv',index=False)
