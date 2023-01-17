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
#Creating our test and train data frames.

df_train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

df_test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")



print(df_train.shape)

print(df_test.shape)
#Importing libraries

import matplotlib.pyplot as plt #Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python

import seaborn as sns #Seaborn is a Python data visualization library based on matplotlib

from sklearn import linear_model #Ordinary least squares Linear Regression
df_train.columns
#Creating a histogram of SalePrice to understand its distribution.

sns.distplot(df_train["SalePrice"])
#Calculate the correlations across the variables

corr_train = df_train.corr()

print(corr_train)
#Change the plot size 

fig, ax = plt.subplots(figsize=(30,1))



#Visualizing a heatmap of the correlation with seaborn

sns.heatmap(corr_train.sort_values(by=["SalePrice"], ascending=False).head(1), annot = True, fmt='.1g')
#Scatter plot for TotalBsmtSF against SalePrice

plt.scatter(df_train["TotalBsmtSF"], df_train["SalePrice"])

plt.xlabel("TotalBsmtSF")

plt.ylabel("SalePrice")

plt.title("SalePrice for TotalBsmtSF")
#Scatter plot for 1stFlrSF against SalePrice

plt.scatter(df_train["1stFlrSF"], df_train["SalePrice"])

plt.xlabel("1stFlrSF")

plt.ylabel("SalePrice")

plt.title("SalePrice for 1stFlrSF")
#Scatter plot for GrLivArea against SalePrice

plt.scatter(df_train["GrLivArea"], df_train["SalePrice"])

plt.xlabel("GrLivArea")

plt.ylabel("SalePrice")

plt.title("SalePrice for GrLivArea")
#Scatter plot for GarageArea against SalePrice

plt.scatter(df_train["GarageArea"], df_train["SalePrice"])

plt.xlabel("GarageArea")

plt.ylabel("SalePrice")

plt.title("SalePrice for GarageArea")
#Set the graph size

fig, ax = plt.subplots(figsize=(15,15))



#Create a boxplot of GarageArea for GarageCars

sns.boxplot(x=df_train["GarageCars"], y=df_train["GarageArea"]).set_title("GarageArea for GarageCars")
#Set the graph size

fig, ax = plt.subplots(figsize=(15,15))



#Create a boxplot of SalePrice for OverallQual

sns.boxplot(x=df_train["OverallQual"], y=df_train["SalePrice"]).set_title("SalePrice for OverallQual")
#Set the graph size

fig, ax = plt.subplots(figsize=(15,15))



#Create a boxplot of SalePrice for FullBath

sns.boxplot(x=df_train["FullBath"], y=df_train["SalePrice"]).set_title("SalePrice for FullBath")
#Set the graph size

fig, ax = plt.subplots(figsize=(15,15))



#Create a boxplot of SalePrice for GarageCars

sns.boxplot(x=df_train["GarageCars"], y=df_train["SalePrice"]).set_title("SalePrice for GarageCars")
#Combining the train and test dataframes into one

df_both = pd.concat((df_train, df_test)).reset_index(drop=True)

df_both.drop(["SalePrice"], axis=1, inplace=True)

df_both.shape
#Cheking for missing values

df_both_null = df_both.isnull().sum() / len(df_both) * 100



#Removing all variables where there are no missing values

df_both_null = df_both_null[df_both_null != 0].sort_values(ascending=False)

print(df_both_null)
#Replacing "NA" with "None" for PoolQC

df_both["PoolQC"] = df_both["PoolQC"].fillna("None")
#Replacing the null variables with None

for i in ("MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "MasVnrType", "MSSubClass"):

    df_both[i] = df_both[i].fillna("None")





df_both_null = df_both.isnull().sum() / len(df_both) * 100



df_both_null = df_both_null[df_both_null != 0].sort_values(ascending=False)

print(df_both_null)

sns.distplot(df_train["LotFrontage"])
#Replacing null values with the median

df_both["LotFrontage"] = df_both["LotFrontage"].fillna(df_train["LotFrontage"].median())



df_both_null = df_both.isnull().sum() / len(df_both) * 100



df_both_null = df_both_null[df_both_null != 0].sort_values(ascending=False)

print(df_both_null)
#Replacing null values with 0

for i in ("GarageYrBlt", "GarageArea", "GarageCars", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath", "MasVnrArea"):

    df_both[i] = df_both[i].fillna(0)

    

df_both_null = df_both.isnull().sum() / len(df_both) * 100



df_both_null = df_both_null[df_both_null != 0].sort_values(ascending=False)

print(df_both_null)
#Print the value frequency for our remaining missing data columns

for i in ("MSZoning", "Functional", "Utilities", "SaleType", "KitchenQual", "Electrical", "Exterior2nd", "Exterior1st"):

    print(df_both[i].value_counts())

#Replacing null values with the most common value

df_both["MSZoning"] = df_both["MSZoning"].fillna("RL")

df_both["Functional"] = df_both["Functional"].fillna("Typ")

df_both["Utilities"] = df_both["Utilities"].fillna("AllPub")

df_both["SaleType"] = df_both["SaleType"].fillna("WD")

df_both["Electrical"] = df_both["Electrical"].fillna("SBrkr")

df_both["Exterior2nd"] = df_both["Exterior2nd"].fillna("VinylSd")

df_both["Exterior1st"] = df_both["Exterior1st"].fillna("VinylSd")



df_both_null = df_both.isnull().sum() / len(df_both) * 100



df_both_null = df_both_null[df_both_null != 0].sort_values(ascending=False)

print(df_both_null)
#Removing the row with the missing value

KitchenQual_df = df_both.dropna(axis=0)

KitchenQual_df.shape



#Creating a dummy column where the categorical KitchenQual values are placed with dummy numerical ones so we can train our model

print(KitchenQual_df.dtypes)

KitchenQual_df["dummy_KitchenQual"] = KitchenQual_df["KitchenQual"].map({"Ex": 4, "Gd": 3, "TA": 2, "Fa": 1, "Po": 0})

print(KitchenQual_df.dtypes)
corr_KitchenQual = KitchenQual_df.corr()

fig, ax = plt.subplots(figsize=(30,1))

sns.heatmap(corr_KitchenQual.sort_values(by=["dummy_KitchenQual"], ascending=False).head(1), annot = True, fmt='.1g')
print(df_both[["OverallQual", "YearBuilt", "YearRemodAdd", "GarageCars", "GarageArea", "KitchenQual"]].loc[[1555]])
from sklearn import linear_model



KitchenQual_X = KitchenQual_df[["OverallQual", "YearBuilt", "YearRemodAdd", "GarageCars", "GarageArea"]]

KitchenQual_Y = KitchenQual_df["dummy_KitchenQual"]



regr_KitchenQual = linear_model.LinearRegression()

regr_KitchenQual.fit(KitchenQual_X, KitchenQual_Y)





print("Predicted missing KitchenQual value: " + str(regr_KitchenQual.predict(df_both[["OverallQual", "YearBuilt", "YearRemodAdd", "GarageCars", "GarageArea"]].loc[[1555]])))
df_both["KitchenQual"] = df_both["KitchenQual"].fillna("TA")



df_both_null = df_both.isnull().sum() / len(df_both) * 100



df_both_null = df_both_null[df_both_null != 0].sort_values(ascending=False)

print(df_both_null)

df_both['TotalSF'] = df_both['TotalBsmtSF'] + df_both['1stFlrSF'] + df_both['2ndFlrSF']
df_both.columns
#Variable which have data types object

var_cat = df_both.dtypes[df_both.dtypes==object].index.values.tolist()

#Dataframe with object datatype variables

df_cat = df_both[var_cat]

df_cat.head(5)


print(list(df_cat.columns))

print("\n\n")



df_cat_dummies = pd.get_dummies(df_cat)



#from sklearn.preprocessing import OneHotEncoder

#encode = OneHotEncoder(drop='first',sparse=False)

#encode.fit(df_cat)



#df_cat_dummies = encode.transform(df_cat)

#df_cat_dummies = pd.DataFrame(df_cat_dummies, columns=encode.get_feature_names(), index=df_cat.index)

print(list(df_cat_dummies.columns))
df_cat_dummies.head(10)
print(f"The shape of combined: {df_both.shape}")

print(f"The shape of categorical variable: {df_cat.shape}")

print(f"The shape of categoricall dummy variables: {df_cat_dummies.shape}")
print(df_both.columns)
df_all_variables = df_both.join(df_cat_dummies)

df_all_variables = df_all_variables.drop(df_cat, axis=1)

df_all_variables.shape

print(list(df_all_variables.columns))
ntrain = df_train.shape[0]

ntest = df_test.shape[0]

print(ntrain, ntest)
print(df_train.shape, df_test.shape)

df_train_final = df_all_variables[:ntrain]

df_test_final = df_all_variables[ntrain:]

df_train_final["SalePrice"] = df_train["SalePrice"]



print(df_train_final.shape, df_test_final.shape)
X_train = df_train_final.drop('SalePrice', axis=1)

y_train = df_train_final["SalePrice"]

X_test = df_test_final



print(f"X_train shape: {X_train.shape}")

print(f"X_test shape: {X_test.shape}")

print(f"y_train shape: {y_train.shape}")



regr = linear_model.LinearRegression()

regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)



final_submission = pd.DataFrame({

        "Id": X_test["Id"],

        "SalePrice": y_pred

    })



final_submission.to_csv("final_submission.csv", index=False)

final_submission.head()