import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score
train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")



train_df = train.drop(columns=["Id"])
#train_df.isnull().sum().to_frame("nulls").sort_values("nulls",ascending=False).head(20)
train_df = train_df.drop(columns=["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu"])

train_df = train_df.drop(columns=["Exterior1st", "GarageYrBlt", "MSZoning", "HouseStyle", "Electrical", "GarageQual"])



train_df["LotFrontage"] = train_df["LotFrontage"].fillna(0)

train_df["MasVnrArea"] = train_df["MasVnrArea"].fillna(0)

train_df["GarageType"] = train_df["GarageType"].fillna("No Garage")

train_df["GarageCond"] = train_df["GarageCond"].fillna("No Garage")

#train_df["GarageYrBlt"] = train_df["GarageYrBlt"].fillna("No Garage")

train_df["GarageFinish"] = train_df["GarageFinish"].fillna("No Garage")

#train_df["GarageQual"] = train_df["GarageQual"].fillna("No Garage")

train_df["BsmtExposure"] = train_df["BsmtExposure"].fillna("No Basement")

train_df["BsmtFinType2"] = train_df["BsmtFinType2"].fillna("No Basement")

train_df["BsmtCond"] = train_df["BsmtCond"].fillna("No Basement")

train_df["BsmtQual"] = train_df["BsmtQual"].fillna("No Basement")

train_df["BsmtFinType1"] = train_df["BsmtFinType1"].fillna("No Basement")

train_df["MasVnrType"] = train_df["MasVnrType"].fillna("No Area")

#train_df["Electrical"] = train_df["Electrical"].fillna("No Electricity")
train_df.isnull().sum().to_frame("nulls").sort_values("nulls",ascending=False).head(20)
corr_matrix_num = train_df.corr()

top_corr_num_features = corr_matrix_num.index[abs(corr_matrix_num["SalePrice"]) > 0.4]

top_corr_num_features = [x for x in top_corr_num_features if x != "SalePrice"]
plt.figure(figsize=(10,10))

g = sns.heatmap(train_df[top_corr_num_features].corr(),annot=True,cmap="RdYlGn")
sns.barplot(train_df["OverallQual"], train_df["SalePrice"])
train_df.select_dtypes(include=["object"]).nunique().to_frame("UniqueValuesCount").sort_values("UniqueValuesCount",ascending=False)
cat_list = train_df.select_dtypes(include=["object"]).columns.to_list()
dummies_df = pd.get_dummies(train_df[cat_list])

dummies_df["SalePrice"] = train_df["SalePrice"]
corr_matrix_cat = dummies_df.corr()

top_corr_cat_features = corr_matrix_cat.index[abs(corr_matrix_cat["SalePrice"]) > 0.2]

top_corr_cat_features = [x for x in top_corr_cat_features if x != "SalePrice"]
top_corr_cat_features = [x.split('_')[0] for x in top_corr_cat_features]

top_corr_cat_features = list(dict.fromkeys(top_corr_cat_features))
dataset = train_df[top_corr_num_features + top_corr_cat_features]
dataset = train_df[top_corr_num_features + top_corr_cat_features]

dataset = pd.get_dummies(dataset, columns=top_corr_cat_features)

dataset["SalePrice"] = train_df["SalePrice"]
X = dataset.iloc[:, dataset.columns != "SalePrice"].values

y = dataset.iloc[:, dataset.columns == "SalePrice"].values

#y = dummies_df["SalePrice"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.01, random_state = 0)
scaler_X = StandardScaler()

X_train = scaler_X.fit_transform(X_train)

X_test = scaler_X.transform(X_test)
regressor = LinearRegression()

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print("mean_squared_error: ", mean_squared_error(y_test, y_pred), "\nr2_score: ", r2_score(y_test, y_pred))
test_df = test.drop(columns=["Id"])
test_df = test_df.drop(columns=["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu"])

test_df = test_df.drop(columns=["Exterior1st", "GarageYrBlt", "MSZoning", "HouseStyle", "Electrical", "GarageQual"])



test_df["LotFrontage"] = test_df["LotFrontage"].fillna(0)

test_df["MasVnrArea"] = test_df["MasVnrArea"].fillna(0)

test_df["GarageArea"] = test_df["GarageArea"].fillna(0)

test_df["GarageCars"] = test_df["GarageCars"].fillna(0)

test_df["GarageType"] = test_df["GarageType"].fillna("No Garage")

test_df["GarageCond"] = test_df["GarageCond"].fillna("No Garage")

#test_df["GarageYrBlt"] = test_df["GarageYrBlt"].fillna("No Garage")

test_df["GarageFinish"] = test_df["GarageFinish"].fillna("No Garage")

#test_df["GarageQual"] = test_df["GarageQual"].fillna("No Garage")

test_df["BsmtExposure"] = test_df["BsmtExposure"].fillna("No Basement")

test_df["BsmtFinType2"] = test_df["BsmtFinType2"].fillna("No Basement")

test_df["BsmtCond"] = test_df["BsmtCond"].fillna("No Basement")

test_df["BsmtQual"] = test_df["BsmtQual"].fillna("No Basement")

test_df["BsmtFinType1"] = test_df["BsmtFinType1"].fillna("No Basement")

test_df["MasVnrType"] = test_df["MasVnrType"].fillna("No Area")

#test_df["Electrical"] = test_df["Electrical"].fillna("No Electricity")



#test_df["MSZoning"] = test_df["MSZoning"].fillna("A")

test_df["SaleType"] = test_df["SaleType"].fillna("Oth")

test_df["TotalBsmtSF"] = test_df["TotalBsmtSF"].fillna(0)

test_df["KitchenQual"] = test_df["KitchenQual"].fillna("TA")

#test_df["Exterior1st"] = test_df["Exterior1st"].fillna("Other")

test_df["Exterior2nd"] = test_df["Exterior2nd"].fillna("Other")
test_df.isnull().sum().to_frame("nulls").sort_values("nulls",ascending=False).head(20)
test_dataset = test_df[top_corr_num_features + top_corr_cat_features]

test_dataset = pd.get_dummies(test_dataset, columns=top_corr_cat_features)
test_dataset.isnull().sum().to_frame("nulls").sort_values("nulls",ascending=False).head(20)
test_dataset
[x for x in dataset.columns if x not in test_dataset.columns]
X_t = test_dataset.values
X_t
X_t = scaler_X.transform(X_t)
y_pred = regressor.predict(X_t)
y_pred_list = [x for x in y_pred]
y_final = np.concatenate(y_pred_list, axis=0)
submission_lr = pd.DataFrame({

        "Id": test["Id"],

        "SalePrice": y_final

})



submission_lr.to_csv('submission_lr.csv', index=False)