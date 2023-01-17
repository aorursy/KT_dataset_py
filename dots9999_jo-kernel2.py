import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing
# Fix random seed for reproducibility

np.random.seed(10)

# Set figure size

plt.rcParams["figure.figsize"]=(10,5)
# Train Data

train=pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
#Test Data

test=pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
#DATA SHAPE
train.shape
test.shape
train.info()
# MISSING VALUES
train.isnull().sum()
test.isnull().sum()
train.fillna(method="bfill", inplace=True)

test.fillna(method="bfill", inplace=True)
train.fillna(method="ffill", inplace=True)

test.fillna(method="ffill", inplace=True)
#Check remaining NA values

train.info()
train.head()
test.head()
# COLUMNS IN THE DATA

train.columns
label_encode=preprocessing.LabelEncoder()
columns=['MSSubClass','MSZoning','Street','Alley', 'LotShape', 'LandContour', 'Utilities','LotConfig',

           'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType','HouseStyle','OverallQual', 

            'OverallCond','RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd','MasVnrType','ExterQual', 

            'ExterCond', 'Foundation', 'BsmtQual','BsmtCond', 'BsmtExposure', 'BsmtFinType1','BsmtFinType2',

            'Heating','HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual','Functional','FireplaceQu', 'GarageType',

            'GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC','Fence', 'MiscFeature','SaleType','SaleCondition']
# Changing data type to category

for i in columns:

    train[i]=train[i].astype("category")

    test[i]=test[i].astype("category")
# Counting and plotting counts of attributes

for i in columns:

    plt.title(i)

    sns.countplot(i,data=train)

    plt.xticks(rotation=60)

    plt.show()
# Label encoding for test and train data

for i in columns:

    train[i]=label_encode.fit_transform(train[i])

    test[i]=label_encode.fit_transform(test[i])
train
test
train.describe()
#Correlation of Attributes of the data

correlation=train.corr(method="spearman")

correlation
#Heatmap of correlation between features in train data

correlation.style.background_gradient(cmap='viridis').set_precision(2)
for column in columns:

    plt.title(column)

    sns.scatterplot(column,"SalePrice",data=train)

    plt.show()
# Distribution plot for sales

sns.distplot(train["SalePrice"])

plt.title("SALE PRICE DISTRIBUTION PLOT")

plt.ylabel("Density")

plt.xlabel("Sale Price")

plt.xticks(rotation=90)

plt.show()
x_train=pd.DataFrame(train[['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',

       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',

       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',

       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',

       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',

       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',

       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',

       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',

       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',

       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',

       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',

       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',

       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',

       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',

       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',

       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',

       'SaleCondition']])

y_train=pd.DataFrame(train.iloc[:,-1])
x_test=pd.DataFrame(test[['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',

       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',

       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',

       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',

       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',

       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',

       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',

       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',

       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',

       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',

       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',

       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',

       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',

       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',

       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',

       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',

       'SaleCondition']])
#FIT AND PREDICT SALE PRICE
# Import and use random forest

from sklearn.ensemble import RandomForestRegressor as RF
# Model Fit and Predict

regressor = RF(n_estimators=100,

             criterion='mse',

             max_features= None, 

             max_depth = 14,bootstrap=True)

regressor=regressor.fit(x_train, y_train.values.ravel())

regressor.fit(x_train, y_train.values.ravel())

#Predict values

y_pred = regressor.predict(x_test)
# Make predicted values as DataFrame

y_pred=pd.DataFrame(y_pred)
y_pred.rename(columns={0:"SalePrice"},inplace=True)
y_pred["Id"]=test["Id"]
# Values for submission

final=y_pred[["Id","SalePrice"]]
final.head()
#Submission

final.to_csv("PriceSubmission.csv",index=False)