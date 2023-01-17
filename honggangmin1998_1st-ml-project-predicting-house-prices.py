import pandas as pd

import numpy as np



from matplotlib import pyplot as plt

import seaborn as sns



train_df = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test_df = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")



print(train_df.shape)

print(test_df.shape)
train_df.head()
test_df.head()
train_df.info()
test_df.info()
sns.distplot(train_df["SalePrice"])
sns.boxplot(data = train_df, x = "MSSubClass", y = "SalePrice")
sns.boxplot(data = train_df, x = "MSZoning", y = "SalePrice")
sns.scatterplot(data = train_df, x = "LotFrontage", y = "SalePrice")
sns.scatterplot(data = train_df, x = "LotArea", y = "SalePrice")
sns.boxplot(data = train_df, x = "Street", y = "SalePrice")
sns.boxplot(data = train_df, x = "Alley", y = "SalePrice")
no_alley = train_df[train_df["Alley"].isnull()]

plt.xlabel("NA (No alley access)")

sns.boxplot(y = no_alley["SalePrice"])
sns.boxplot(data = train_df, x = "LotShape", y = "SalePrice")
sns.boxplot(data = train_df, x = "LandContour", y = "SalePrice")
sns.boxplot(data = train_df, x = "Utilities", y = "SalePrice")
train_df["Utilities"].value_counts()
test_df["Utilities"].describe()
sns.boxplot(data = train_df, x = "LotConfig", y = "SalePrice")
sns.boxplot(data = train_df, x = "LandSlope", y = "SalePrice")
plt.figure(figsize = (12, 6))

plt.xticks(rotation = 45)

sns.boxplot(data = train_df, x = "Neighborhood", y = "SalePrice")
median_by_group = train_df.groupby(by = ["Neighborhood"])["SalePrice"].median().sort_values(ascending = False).index

plt.figure(figsize = (12, 6))

plt.xticks(rotation = 45)

sns.boxplot(data = train_df, x = "Neighborhood", y = "SalePrice", order = median_by_group)
sns.boxplot(data = train_df, x = "Condition1", y = "SalePrice")
sns.boxplot(data = train_df, x = "Condition2", y = "SalePrice")
sns.boxplot(data = train_df, x = "BldgType", y = "SalePrice")
sns.boxplot(data = train_df, x = "HouseStyle", y = "SalePrice")
sns.boxplot(data = train_df, x = "OverallQual", y = "SalePrice")
sns.boxplot(data = train_df, x = "OverallCond", y = "SalePrice")
sns.scatterplot(data = train_df, x = "YearBuilt", y = "SalePrice")
sns.scatterplot(data = train_df, x = "YearRemodAdd", y = "SalePrice")
sns.boxplot(data = train_df, x = "RoofStyle", y = "SalePrice")
plt.xticks(rotation = 30)

sns.boxplot(data = train_df, x = "RoofMatl", y = "SalePrice")
plt.xticks(rotation = 45)

sns.boxplot(data = train_df, x = "Exterior1st", y = "SalePrice")
plt.xticks(rotation = 45)

sns.boxplot(data = train_df, x = "Exterior2nd", y = "SalePrice")
sns.boxplot(data = train_df, x = "MasVnrType", y = "SalePrice")
sns.scatterplot(data = train_df, x = "MasVnrArea", y = "SalePrice")
sns.boxplot(data = train_df, x = "ExterQual", y = "SalePrice")

train_df["ExterQual"].value_counts()
test_df.info()

test_df["ExterQual"].value_counts()
sns.boxplot(data = train_df, x = "ExterCond", y = "SalePrice")
sns.boxplot(data = train_df, x = "Foundation", y = "SalePrice")
sns.boxplot(data = train_df, x = "BsmtQual", y = "SalePrice")

train_df["BsmtQual"].value_counts()
sns.boxplot(data = train_df, x = "BsmtCond", y = "SalePrice")

train_df["BsmtCond"].value_counts()
sns.boxplot(data = train_df, x = "BsmtExposure", y = "SalePrice")
sns.boxplot(data = train_df, x = "BsmtFinType1", y = "SalePrice")
sns.scatterplot(data = train_df, x = "BsmtFinSF1", y = "SalePrice")
sns.boxplot(data = train_df, x = "BsmtFinType2", y = "SalePrice")
sns.scatterplot(data = train_df, x = "BsmtFinSF2", y = "SalePrice")
sns.scatterplot(data = train_df, x = "BsmtUnfSF", y = "SalePrice")
sns.scatterplot(data = train_df, x = "TotalBsmtSF", y = "SalePrice")
sns.boxplot(data = train_df, x = "Heating", y = "SalePrice")
sns.boxplot(data = train_df, x = "HeatingQC", y = "SalePrice")
sns.boxplot(data = train_df, x = "CentralAir", y = "SalePrice")
sns.boxplot(data = train_df, x = "Electrical", y = "SalePrice")
sns.scatterplot(data = train_df, x = "1stFlrSF", y = "SalePrice")
sns.scatterplot(data = train_df, x = "2ndFlrSF", y = "SalePrice")
sns.scatterplot(data = train_df, x = "LowQualFinSF", y = "SalePrice")
sns.scatterplot(data = train_df, x = "GrLivArea", y = "SalePrice")
sns.scatterplot(data = train_df, x = "BsmtFullBath", y = "SalePrice")
sns.scatterplot(data = train_df, x = "BsmtHalfBath", y = "SalePrice")
sns.scatterplot(data = train_df, x = "FullBath", y = "SalePrice")
sns.scatterplot(data = train_df, x = "HalfBath", y = "SalePrice")
sns.scatterplot(data = train_df, x = "BedroomAbvGr", y = "SalePrice")
sns.scatterplot(data = train_df, x = "KitchenAbvGr", y = "SalePrice")
sns.boxplot(data = train_df, x = "KitchenQual", y = "SalePrice")

train_df["KitchenQual"].value_counts()
test_df["KitchenQual"].value_counts()
sns.boxplot(data = train_df, x = "TotRmsAbvGrd", y = "SalePrice")
sns.boxplot(data = train_df, x = "Functional", y = "SalePrice")
sns.scatterplot(data = train_df, x = "Fireplaces", y = "SalePrice")
sns.boxplot(data = train_df, x = "FireplaceQu", y = "SalePrice")
sns.boxplot(data = train_df, x = "GarageType", y = "SalePrice")
sns.scatterplot(data = train_df, x = "GarageYrBlt", y = "SalePrice")
sns.boxplot(data = train_df, x = "GarageFinish", y = "SalePrice")
sns.scatterplot(data = train_df, x = "GarageCars", y = "SalePrice")
sns.scatterplot(data = train_df, x = "GarageArea", y = "SalePrice")
sns.boxplot(data = train_df, x = "GarageQual", y = "SalePrice")
sns.boxplot(data = train_df, x = "GarageCond", y = "SalePrice")
sns.boxplot(data = train_df, x = "PavedDrive", y = "SalePrice")
sns.scatterplot(data = train_df, x = "WoodDeckSF", y = "SalePrice")
sns.scatterplot(data = train_df, x = "OpenPorchSF", y = "SalePrice")
sns.scatterplot(data = train_df, x = "EnclosedPorch", y = "SalePrice")
sns.scatterplot(data = train_df, x = "3SsnPorch", y = "SalePrice")
sns.scatterplot(data = train_df, x = "ScreenPorch", y = "SalePrice")
sns.scatterplot(data = train_df, x = "PoolArea", y = "SalePrice")
sns.boxplot(data = train_df, x = "PoolQC", y = "SalePrice")
sns.boxplot(data = train_df, x = "Fence", y = "SalePrice")
sns.boxplot(data = train_df, x = "MiscFeature", y = "SalePrice")

train_df["MiscFeature"].value_counts()
sns.scatterplot(data = train_df, x = "MiscVal", y = "SalePrice")
sns.scatterplot(data = train_df, x = "MoSold", y = "SalePrice")
sns.scatterplot(data = train_df, x = "YrSold", y = "SalePrice")
sns.boxplot(data = train_df, x = "SaleType", y = "SalePrice")
sns.boxplot(data = train_df, x = "SaleCondition", y = "SalePrice")
from sklearn import preprocessing



var_list = ["Street","Neighborhood","ExterQual","KitchenQual","CentralAir"]



le = preprocessing.LabelEncoder()



train_df1 = train_df.copy()



for column in var_list:

    train_df1[column] = le.fit_transform(train_df1[column])



var_list = ["LotFrontage","OverallQual","TotalBsmtSF","GrLivArea","YearBuilt","Street","Neighborhood","ExterQual","KitchenQual","CentralAir","SalePrice"]



plt.figure(figsize = (10, 10))

sns.heatmap(train_df1[var_list].corr(), annot = True, cmap="binary")
sns.distplot(train_df["LotFrontage"])
sns.distplot(test_df["LotFrontage"])
train_df["LotFrontage"].fillna(train_df["LotFrontage"].median(), inplace = True)

test_df["LotFrontage"].fillna(test_df["LotFrontage"].median(), inplace = True)
sns.distplot(test_df["TotalBsmtSF"])
test_df["TotalBsmtSF"].fillna(test_df["TotalBsmtSF"].median(), inplace = True)
test_df["KitchenQual"].value_counts()

test_df["KitchenQual"].fillna("TA", inplace = True)
full = pd.concat([train_df, test_df], ignore_index = True)

full
from sklearn.preprocessing import OneHotEncoder



encoder=OneHotEncoder(sparse = False)



full1 = full.copy()



def onehotencoding(df, column_name):

    full1_encoded = pd.DataFrame(encoder.fit_transform(full1[[column_name]]))

    full1_encoded.columns = encoder.get_feature_names([column_name])

    full1.drop([column_name] ,axis=1, inplace=True)

    final = pd.concat([full1, full1_encoded], axis=1)

    return final



full1 = onehotencoding(full1, "ExterQual")

full1 = onehotencoding(full1, "KitchenQual")

full1
final_train_df = full1[:len(train_df)]

final_test_df = full1[len(train_df):]

column_list = ["LotFrontage","OverallQual","TotalBsmtSF","GrLivArea","YearBuilt",

               "ExterQual_Ex","ExterQual_Fa","ExterQual_Gd","ExterQual_TA",

               "KitchenQual_Ex","KitchenQual_Fa","KitchenQual_Gd","KitchenQual_TA"]



X_train_final = final_train_df[column_list]

y_train_final = final_train_df["SalePrice"]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_train_final, y_train_final)
from sklearn.neighbors import KNeighborsRegressor



knr = KNeighborsRegressor()



knr.fit(X_train, y_train)



print("The R^2 for the training set is {}".format(knr.score(X_train, y_train)))

print("The R^2 for the testing set is {}".format(knr.score(X_test, y_test)))
from sklearn.linear_model import LinearRegression



linreg = LinearRegression()

linreg.fit(X_train, y_train)



print("The R^2 for the training set is: {}".format(linreg.score(X_train, y_train)))

print("The R^2 for the testing set is: {}".format(linreg.score(X_test, y_test)))
from sklearn.linear_model import Ridge



ridge = Ridge()

ridge.fit(X_train, y_train)



print("Training set score: {}".format(ridge.score(X_train, y_train)))

print("Testing set score: {}".format(ridge.score(X_test, y_test)))
from sklearn.linear_model import Lasso



lasso = Lasso()

lasso.fit(X_train, y_train)



print("Training set score: {}".format(lasso.score(X_train, y_train)))

print("Testing set score: {}".format(lasso.score(X_test, y_test)))

print("Number of features used: {}".format(np.sum(lasso.coef_ != 0)))
prediction = ridge.predict(final_test_df[column_list])

prediction = pd.DataFrame(prediction)

final_table = pd.concat([test_df["Id"], prediction], axis = 1)

final_table.columns = ["Id", "SalePrice"]

final_table
final_table.to_csv("submission.csv", index = False)