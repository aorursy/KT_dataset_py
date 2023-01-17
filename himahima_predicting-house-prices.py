import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('white')
%matplotlib inline

df = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
df.head()
df.info()

df.columns
print(df.columns.values)
df.describe()
df.describe(include=['O'])
f, axes = plt.subplots(ncols=4, figsize=(14,4))
# Lot Area: In Square Feet
sns.distplot(df['LotArea'], kde=False, color="#DF3A01", ax=axes[0]).set_title("Distribution of LotArea")
axes[0].set_ylabel("Square Ft")
axes[0].set_xlabel("Amount of Houses")

# MoSold: Year of the Month sold
sns.distplot(df['MoSold'], kde=False, color="#045FB4", ax=axes[1]).set_title("Monthly Sales Distribution")
axes[1].set_ylabel("Amount of Houses Sold")
axes[1].set_xlabel("Month of the Year")

# House Value
sns.distplot(df['SalePrice'], kde=False, color="#088A4B", ax=axes[2]).set_title("Monthly Sales Distribution")
axes[2].set_ylabel("Number of Houses ")
axes[2].set_xlabel("Price of the House")

# YrSold: Year the house was sold.
sns.distplot(df['YrSold'], kde=False, color="#FE2E64", ax=axes[3]).set_title("Year Sold")
axes[3].set_ylabel("Number of Houses ")
axes[3].set_xlabel("Year Sold")

plt.show()
plt.figure(figsize=(10,6))
sns.distplot(df['SalePrice'], color='r')
plt.title('Distribution of Sales Price', fontsize=18)

plt.show()
# People tend to move during the summer
plt.figure(figsize=(12,8))
sns.countplot(y="MoSold", hue="YrSold", data=df)
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(x='YrSold', y='SalePrice', data=df)
plt.xlabel('Year Sold', fontsize=14)
plt.ylabel('Price sold', fontsize=14)
plt.title('Houses Sold per Year', fontsize=16)
plt.figure(figsize=(10,6))
plt.scatter(x=df['GrLivArea'], y=df['SalePrice'], color='c', alpha=0.5)
plt.xlabel('GrLivArea', fontsize=12)
plt.ylabel('Sale price',  fontsize=12)
plt.show()
plt.figure(figsize=(10,6))
df = df.drop(df[(df['GrLivArea']>4000)& (df['SalePrice']>7000)].index)
plt.scatter(x='GrLivArea', y='SalePrice', data=df,color='c', alpha=0.5)
plt.xlabel('GrLivArea', fontsize=12)
plt.ylabel('Sale price',  fontsize=12)
plt.figure(figsize=(10,6))
plt.scatter(x=df['TotalBsmtSF'], y=df['SalePrice'], color='b', alpha=0.5)
plt.xlabel('TotalBsmtSF', fontsize=12)
plt.ylabel('Sale price', fontsize=12)
plt.show()
plt.figure(figsize=(10,6))
df = df.drop(df[(df['TotalBsmtSF']>2900)& (df['SalePrice']>5500)].index)
plt.scatter(x='TotalBsmtSF', y='SalePrice', data=df,color='b', alpha=0.5)
plt.xlabel('TotalBsmtSF', fontsize=12)
plt.ylabel('Sale price',  fontsize=12)
a = 'OverallQual'
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=a, y="SalePrice", data=df)
fig.axis(ymin=0, ymax=800000)
a = 'YearBuilt'
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=a, y="SalePrice", data=df)
fig.axis(ymin=0, ymax=800000)
corrmat = df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
df["MiscFeature"] = df["MiscFeature"].fillna("None")
df["Alley"] = df["Alley"].fillna("None")
df["Fence"] = df["Fence"].fillna("None")
df["FireplaceQu"] = df["FireplaceQu"].fillna("None")
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    df[col] = df[col].fillna('None')
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    df[col] = df[col].fillna('None')
df['MSZoning'] = df['MSZoning'].fillna(df['MSZoning'].mode()[0])
df["MasVnrType"] = df["MasVnrType"].fillna("None")
df["Functional"] = df["Functional"].fillna("Typ")
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
df['KitchenQual'] = df['KitchenQual'].fillna(df['KitchenQual'].mode()[0])
df['Exterior1st'] = df['Exterior1st'].fillna(df['Exterior1st'].mode()[0])
df['Exterior2nd'] = df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0])
df['SaleType'] = df['SaleType'].fillna(df['SaleType'].mode()[0])
df['MSSubClass'] = df['MSSubClass'].fillna("None")
df['MasVnrArea'] = df['MasVnrArea'].fillna(df['MasVnrArea'].mode()[0])
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['GarageYrBlt'].mode()[0])
df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].mode()[0])

             
df = df.drop((missing_data[missing_data['Total'] > 1]).index,1)
df = df.drop(df.loc[df['Electrical'].isnull()].index)
df.isnull().sum().max()
df = pd.get_dummies(df)
df.head()

df.drop(['Id'], axis=1, inplace=True)
df.head()
X = df.drop(columns=["SalePrice"])
y = df["SalePrice"]

from sklearn.preprocessing import MinMaxScaler

#MinMaxScaler for Data

scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
X = scaler.fit_transform(X)
df.head()

from sklearn.model_selection import train_test_split

#Splitting data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle =True)

from sklearn.linear_model import Lasso

#Applying Lasso Regression Model 


LassoRegressionModel = Lasso(alpha=1.0,random_state=33,normalize=False)
LassoRegressionModel.fit(X_train, y_train)

#Calculating Details
print('Lasso Regression Train Score is : ' , LassoRegressionModel.score(X_train, y_train))
print('Lasso Regression Test Score is : ' , LassoRegressionModel.score(X_test, y_test))

#print('----------------------------------------------------')

#Calculating Prediction
y_pred = LassoRegressionModel.predict(X_test)
print('Predicted Value for Lasso Regression is : ' , y_pred[:10])
from sklearn.linear_model import Ridge

#Applying Ridge Regression Model 
RidgeRegressionModel = Ridge(alpha=1.0,random_state=33)
RidgeRegressionModel.fit(X_train, y_train)

#Calculating Details
print('Ridge Regression Train Score is : ' , RidgeRegressionModel.score(X_train, y_train))
print('Ridge Regression Test Score is : ' , RidgeRegressionModel.score(X_test, y_test))
#print('----------------------------------------------------')

#Calculating Prediction
y_pred = RidgeRegressionModel.predict(X_test)
print('Predicted Value for Ridge Regression is : ' , y_pred[:10])
from sklearn.metrics import mean_absolute_error 

#Calculating Mean Absolute Error
MAEValue = mean_absolute_error(y_test, y_pred, multioutput='uniform_average') # it can be raw_values
print('Mean Absolute Error Value is : ', MAEValue)