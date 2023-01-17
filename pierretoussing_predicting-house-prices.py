import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

df_test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
df.head()
df.describe()
df.info()
df.shape
sns.heatmap(df.isnull(), yticklabels=False, cbar=False)
columns_with_NAs = df.columns[df.isnull().sum() > 0]

columns_with_NAs
df["Alley"].fillna("None", inplace=True)

df["BsmtQual"].fillna("None", inplace=True)

df["BsmtCond"].fillna("None", inplace=True)

df["BsmtExposure"].fillna("None", inplace=True)

df["BsmtFinType1"].fillna("None", inplace=True)

df["BsmtFinType2"].fillna("None", inplace=True)

df["FireplaceQu"].fillna("None", inplace=True)

df["GarageType"].fillna("None", inplace=True)

df["GarageYrBlt"].fillna("None", inplace=True)

df["GarageFinish"].fillna("None", inplace=True)

df["GarageQual"].fillna("None", inplace=True)

df["GarageCond"].fillna("None", inplace=True)

df["PoolQC"].fillna("None", inplace=True)

df["Fence"].fillna("None", inplace=True)

df["MiscFeature"].fillna("None", inplace=True)
df["LotFrontage"].fillna(df["LotFrontage"].mean(), inplace=True)

df["MasVnrType"].fillna(df["MasVnrType"].mode()[0], inplace=True)

df["MasVnrArea"].fillna(df["MasVnrArea"].mean(), inplace=True)

df["Electrical"].fillna(df["Electrical"].mode()[0], inplace=True)
sns.heatmap(df_test.isnull(), yticklabels=False, cbar=False)
columns_with_NAs = df_test.columns[df_test.isnull().sum() > 0]

df[columns_with_NAs].info()
df_test["MSZoning"].fillna(df_test["MSZoning"].mode()[0], inplace=True)

df_test["LotFrontage"].fillna(df_test["LotFrontage"].mean(), inplace=True)

df_test["Alley"].fillna("None", inplace=True)

df_test["Utilities"].fillna(df_test["Utilities"].mode()[0], inplace=True)

df_test["Exterior1st"].fillna(df_test["Exterior1st"].mode()[0], inplace=True)

df_test["Exterior2nd"].fillna(df_test["Exterior2nd"].mode()[0], inplace=True)

df_test["MasVnrType"].fillna(df_test["MasVnrType"].mode()[0], inplace=True)

df_test["MasVnrArea"].fillna(df_test["MasVnrArea"].mean(), inplace=True)

df_test["BsmtQual"].fillna("None", inplace=True)

df_test["BsmtCond"].fillna("None", inplace=True)

df_test["BsmtExposure"].fillna("None", inplace=True)

df_test["BsmtFinType1"].fillna("None", inplace=True)

df_test["BsmtFinSF1"].fillna(0, inplace=True)

df_test["BsmtFinType2"].fillna("None", inplace=True)

df_test["BsmtFinSF2"].fillna(0, inplace=True)

df_test["BsmtUnfSF"].fillna(0, inplace=True)

df_test["TotalBsmtSF"].fillna(0, inplace=True)

df_test["BsmtFullBath"].fillna(0, inplace=True)

df_test["BsmtHalfBath"].fillna(0, inplace=True)

df_test["KitchenQual"].fillna(df_test["KitchenQual"].mode()[0], inplace=True)

df_test["Functional"].fillna(df_test["Functional"].mode()[0], inplace=True)

df_test["FireplaceQu"].fillna("None", inplace=True)

df_test["GarageType"].fillna("None", inplace=True)

df_test["GarageYrBlt"].fillna("None", inplace=True)

df_test["GarageFinish"].fillna("None", inplace=True)

df_test["GarageCars"].fillna(0, inplace=True)

df_test["GarageArea"].fillna(0, inplace=True)

df_test["GarageQual"].fillna("None", inplace=True)

df_test["GarageCond"].fillna("None", inplace=True)

df_test["PoolQC"].fillna("None", inplace=True)

df_test["Fence"].fillna("None", inplace=True)

df_test["MiscFeature"].fillna("None", inplace=True)

df_test["SaleType"].fillna("None", inplace=True)

df_test.columns[df_test.isnull().sum() > 0]
features = df.select_dtypes(include=["object"]).copy()

categorical_features = features.columns
final_df = pd.concat([df, df_test], axis=0)

final_df.shape
for feature in categorical_features:

		#Encode feature

		encoded_feature = pd.get_dummies(final_df[feature], drop_first=True)



		#Remove old version of feature

		final_df.drop([feature], axis=1, inplace=True)



		#Add encoded version of feature

		final_df = pd.concat([final_df, encoded_feature], axis=1)



final_df.shape
final_df = final_df.loc[:, ~final_df.columns.duplicated()]

final_df.shape
df_train = final_df.iloc[:df.shape[0], :]

df_test = final_df.iloc[df.shape[0]:, :]

df_test.shape
df_test.drop(["SalePrice"], axis=1, inplace=True)
X_train = df_train.drop(["SalePrice"], axis=1)

y_train = df_train["SalePrice"]
from xgboost import XGBRegressor

from sklearn.model_selection import RandomizedSearchCV



regressor = XGBRegressor()
hyperparameter_grid = {

    "max_depth" : [4, 6, 8, 10, 12, 14, 16],

    "min_child_weight" : [1, 2, 3, 4],

    "base_score" : [0.3, 0.4, 0.5, 0.6, 0.7],

    "n_estimators" : [100, 200, 400, 800, 1200, 1600],

    "reg_lambda" : [0.7, 0.8, 0.9, 1],

    "reg_alpha" : [0, 0.1, 0.2, 0.3]

}
random_cv = RandomizedSearchCV(estimator=regressor, param_distributions=hyperparameter_grid, cv=5, n_iter=50, n_jobs=4, scoring="neg_root_mean_squared_error")
random_cv.fit(X_train, y_train)
regressor = random_cv.best_estimator_
predicted_prices = regressor.predict(df_test)
submission = pd.DataFrame({'Id': df_test.Id, 'SalePrice': predicted_prices})



submission.to_csv('submission.csv', index=False)

submission.shape