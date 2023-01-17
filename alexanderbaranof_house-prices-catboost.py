import numpy as np

import pandas as pd



from catboost import CatBoostRegressor
train_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test_df =pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train_df.head(2)
test_df.head(2)
train_df.shape
test_df.shape
train_df.info()
test_df.info()
train_df.hist(bins=50, figsize=(22,16))
y_train = train_df['SalePrice']

X_train = train_df.drop(labels=['SalePrice'], axis=1)



test_ids = test_df['Id']



X_train = X_train.drop(labels=['Id'], axis=1)

test_df = test_df.drop(labels=['Id'], axis=1)
X_train_na = (X_train.isnull().sum() / len(X_train)) * 100

X_train_na = X_train_na.drop(X_train_na[X_train_na == 0].index).sort_values(ascending=False)

missing_data = pd.DataFrame({"Missing Ratio": X_train_na})

missing_data.head()
test_df_na = (test_df.isnull().sum() / len(test_df)) * 100

test_df_na = test_df_na.drop(test_df_na[test_df_na == 0].index).sort_values(ascending=False)

missing_data = pd.DataFrame({"Missing Ratio": test_df_na})

missing_data.head()
y_train = np.log1p(y_train)
X_train["PoolQC"] = X_train["PoolQC"].fillna("None")

test_df["PoolQC"] = test_df["PoolQC"].fillna("None")
X_train["MiscFeature"] = X_train["MiscFeature"].fillna("None")

test_df["MiscFeature"] = test_df["MiscFeature"].fillna("None")
X_train["Alley"] = X_train["Alley"].fillna("None")

test_df["Alley"] = test_df["Alley"].fillna("None")
X_train["Fence"] = X_train["Fence"].fillna("None")

test_df["Fence"] = test_df["Fence"].fillna("None")
X_train["FireplaceQu"] = X_train["FireplaceQu"].fillna("None")

test_df["FireplaceQu"] = test_df["FireplaceQu"].fillna("None")
X_train["LotFrontage"] = X_train.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median())

)

test_df["LotFrontage"] = test_df.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median())

)
for col in ("GarageType", "GarageFinish", "GarageQual", "GarageCond"):

    X_train[col] = X_train[col].fillna("None")

    test_df[col] = test_df[col].fillna("None")
for col in ("GarageYrBlt", "GarageArea", "GarageCars"):

    X_train[col] = X_train[col].fillna(0)

    test_df[col] = test_df[col].fillna(0)
for col in (

    "BsmtFinSF1",

    "BsmtFinSF2",

    "BsmtUnfSF",

    "TotalBsmtSF",

    "BsmtFullBath",

    "BsmtHalfBath",

):

    X_train[col] = X_train[col].fillna(0)

    test_df[col] = test_df[col].fillna(0)
for col in ("BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"):

    X_train[col] = X_train[col].fillna("None")

    test_df[col] = test_df[col].fillna("None")
X_train["MasVnrType"] = X_train["MasVnrType"].fillna("None")

test_df["MasVnrType"] = test_df["MasVnrType"].fillna("None")



X_train["MasVnrArea"] = X_train["MasVnrArea"].fillna(0)

test_df["MasVnrArea"] = test_df["MasVnrArea"].fillna(0)
X_train["MSZoning"] = X_train["MSZoning"].fillna(X_train["MSZoning"].mode()[0])

test_df["MSZoning"] = test_df["MSZoning"].fillna(test_df["MSZoning"].mode()[0])
X_train = X_train.drop(["Utilities"], axis=1)

test_df = test_df.drop(["Utilities"], axis=1)
X_train["Functional"] = X_train["Functional"].fillna("Typ")

test_df["Functional"] = test_df["Functional"].fillna("Typ")
X_train["Electrical"] = X_train["Electrical"].fillna(X_train["Electrical"].mode()[0])

test_df["Electrical"] = test_df["Electrical"].fillna(test_df["Electrical"].mode()[0])
X_train["KitchenQual"] = X_train["KitchenQual"].fillna(X_train["KitchenQual"].mode()[0])

test_df["KitchenQual"] = test_df["KitchenQual"].fillna(test_df["KitchenQual"].mode()[0])
X_train["Exterior1st"] = X_train["Exterior1st"].fillna(X_train["Exterior1st"].mode()[0])

test_df["Exterior2nd"] = test_df["Exterior2nd"].fillna(test_df["Exterior2nd"].mode()[0])
X_train["SaleType"] = X_train["SaleType"].fillna(X_train["SaleType"].mode()[0])

test_df["SaleType"] = test_df["SaleType"].fillna(test_df["SaleType"].mode()[0])
X_train["MSSubClass"] = X_train["MSSubClass"].fillna("None")

test_df["MSSubClass"] = test_df["MSSubClass"].fillna("None")
X_train["Exterior1st"] = X_train["Exterior1st"].fillna(X_train["Exterior1st"].mode()[0])

test_df["Exterior1st"] = test_df["Exterior1st"].fillna(test_df["Exterior1st"].mode()[0])



X_train["Exterior2nd"] = X_train["Exterior2nd"].fillna(X_train["Exterior2nd"].mode()[0])

test_df["Exterior2nd"] = test_df["Exterior2nd"].fillna(test_df["Exterior2nd"].mode()[0])
X_train["MSSubClass"] = X_train["MSSubClass"].apply(str)

test_df["MSSubClass"] = test_df["MSSubClass"].apply(str)
X_train["OverallCond"] = X_train["OverallCond"].astype(str)

test_df["OverallCond"] = test_df["OverallCond"].astype(str)
X_train["YrSold"] = X_train["YrSold"].astype(str)

test_df["YrSold"] = test_df["YrSold"].astype(str)



X_train["MoSold"] = X_train["MoSold"].astype(str)

test_df["MoSold"] = test_df["MoSold"].astype(str)
X_train["TotalSF"] = X_train["TotalBsmtSF"] + X_train["1stFlrSF"] + X_train["2ndFlrSF"]

test_df["TotalSF"] = test_df["TotalBsmtSF"] + test_df["1stFlrSF"] + test_df["2ndFlrSF"]
X_train_na = (X_train.isnull().sum() / len(X_train)) * 100

X_train_na = X_train_na.drop(X_train_na[X_train_na == 0].index).sort_values(ascending=False)

missing_data = pd.DataFrame({"Missing Ratio": X_train_na})

missing_data.head()
test_df_na = (test_df.isnull().sum() / len(test_df)) * 100

test_df_na = test_df_na.drop(test_df_na[test_df_na == 0].index).sort_values(ascending=False)

missing_data = pd.DataFrame({"Missing Ratio": test_df_na})

missing_data.head()
categorical_features_indices = X_train.select_dtypes(include=['object']).columns
categorical_features_indices
X_train.columns
test_df.columns
cbr = CatBoostRegressor(

    eval_metric="RMSE"

)



cbr.fit(

    X_train,

    y_train,

    cat_features=categorical_features_indices

)

y_pred = cbr.predict(test_df[X_train.columns])
y_pred = np.expm1(y_pred)
submission = pd.DataFrame({

        "Id": test_ids,

        "SalePrice": y_pred

    })

submission.to_csv('submission.csv', index=False)


