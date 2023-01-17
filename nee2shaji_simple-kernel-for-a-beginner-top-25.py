#import section

import numpy as np # linear algebra

import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import xgboost as xgb

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

from sklearn.metrics import mean_absolute_error

from scipy.stats import skew

from sklearn.model_selection import cross_val_score, train_test_split

from xgboost import XGBRegressor

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV



#import and understand data

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

#concatenate both train and test data

all_data = pd.concat((df_train, df_test), sort=False).reset_index(drop=True)

#"SalePrice" is the target value. We don't include it in data. We don't want "id" affecting our model. Hence remove it.

all_data.drop(['SalePrice'], axis=1, inplace=True)

all_data = all_data.drop(["Id"], axis=1)
cols_with_missing = [col for col in all_data.columns 

                                 if all_data[col].isnull().any()]

#You can print the cols_with_missing to get better understanding of the columns with missing values

cols_with_missing
#Handle missing values one by one

all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))

all_data["Alley"] = all_data["Alley"].fillna("None")

all_data['Utilities'] = all_data['Utilities'].fillna(all_data['Utilities'].mode()[0])

all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])

all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")

all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    all_data[col] = all_data[col].fillna(0)

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    all_data[col] = all_data[col].fillna('None')

all_data["Electrical"] = all_data["Electrical"].fillna(all_data["Electrical"].mode()[0])

all_data["KitchenQual"] = all_data["KitchenQual"].fillna(all_data["KitchenQual"].mode()[0])

all_data["Functional"] = all_data["Functional"].fillna(all_data["Functional"].mode()[0])

all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")

for col in ('GarageType','GarageFinish','GarageQual','GarageCond'):

    all_data[col] = all_data[col].fillna("None")

for col in ('GarageYrBlt','GarageCars','GarageArea'):

    all_data[col] = all_data[col].fillna(0)

all_data["PoolQC"] = all_data["PoolQC"].fillna("None")

all_data["Fence"] = all_data["Fence"].fillna("None")

all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")

all_data["SaleType"] = all_data["SaleType"].fillna(all_data["SaleType"].mode()[0])



#Now check again if there are anymore columns with missing values.

cols_with_missing = [col for col in all_data.columns 

                                 if all_data[col].isnull().any()]

len(cols_with_missing)
#"SalePrice" is skewed. This isn't good. It is better to apply log transformation.

sns.distplot(df_train['SalePrice']);

df_train["SalePrice"] = np.log1p(df_train["SalePrice"])

sns.distplot(df_train['SalePrice']);

numerical_features = all_data.select_dtypes(exclude = ["object"]).columns

print("Number of numerical features:" + str(len(numerical_features)))



#log transform numerical features

skewness = all_data[numerical_features].apply(lambda x: skew(x))

skewness = skewness[abs(skewness) > 0.7]

skewed_features = skewness.index

all_data[skewed_features] = np.log1p(all_data[skewed_features])
categorical_features = all_data.select_dtypes(include = ["object"]).columns

print("Number of categorical features:" + str(len(categorical_features)))



#getdummies for categorical features

#Create a dataFrame with dummy categorical values

dummy_all_data = pd.get_dummies(all_data[categorical_features])

#Remove categorical features from original data, which leaves original data with only numerical featues

all_data.drop(categorical_features, axis=1, inplace=True)

#Concatenate the numerical features in original data and categorical features with dummies

all_data = pd.concat([all_data, dummy_all_data], axis=1)

#print(all_data.shape)
#Separate training and given test data

X = all_data[:df_train.shape[0]]

test_data = all_data[df_train.shape[0]:]

y = df_train["SalePrice"]
def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv = 5))    #squared mean error

    return(rmse)
m_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X, y)

rmse_cv(m_lasso).mean()   
m_xgb = xgb.XGBRegressor(n_estimators=10000, max_depth=5,learning_rate=0.07)

m_xgb.fit(X, y)
p_xgb = np.expm1(m_xgb.predict(test_data))

p_lasso = np.expm1(m_lasso.predict(test_data))

predicted_prices = 0.75*p_lasso + 0.25*p_xgb

print(predicted_prices)
my_submission = pd.DataFrame({'Id': df_test.Id, 'SalePrice': predicted_prices})

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)