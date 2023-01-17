#Importing all the required Packages

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from scipy import stats
#By this Output will show maximum these numbers of columns and rows

pd.set_option('display.max_column',80)

pd.set_option('display.max_rows',10)
#Importing the Data

data = pd.read_csv('../input/train.csv')

#print(data.head(5))

df_train = pd.DataFrame(data)
#It will Print the list of all the Columns

print(df_train.columns)
#descriptive statistics summary

df_train['SalePrice'].describe()
#histogram

sns.distplot(df_train['SalePrice']);
plt.scatter(df_train['GrLivArea'],df_train['SalePrice'])

plt.show()
plt.scatter(df_train['TotalBsmtSF'],df_train['SalePrice'])

plt.xlabel('Total Basement Surface')

plt.ylabel('SalePrice')

plt.title('Relation b/w Selling Price and Total Basement Area',fontsize = 16)

plt.show()
sns.boxplot(x=df_train['OverallQual'] ,y= df_train['SalePrice'])

plt.title('Relation b/w the Selling Price and Overall Quality of the House',fontsize = 14)

plt.show()
plt.figure(figsize = (25,16))

sns.boxplot(x= "YearBuilt", y="SalePrice", data=df_train)

plt.title('YearBuilt and Selling Price Relation',fontsize = 22)

plt.xticks(rotation=70);

plt.xlabel('YearBuilt',fontsize = 20)

plt.ylabel('SalePrice',fontsize = 20)

plt.show()
#missing data

total = df_train.isnull().sum().sort_values(ascending=False)

percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
#dealing with missing data

df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)

df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)

df_train.isnull().sum().max() #just checking that there's no missing data missing...
print(df_train.columns)
#saleprice correlation matrix 

plt.figure(figsize = (25,16))

cm = sns.heatmap(df_train[['MSSubClass', 'MSZoning', 'LotArea', 'Street', 'LotShape',

       'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood',

       'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual',

       'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl',

       'Exterior1st', 'Exterior2nd', 'ExterQual', 'ExterCond', 'Foundation',

       'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',

       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',

       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',

       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',

       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'GarageCars', 'GarageArea',

       'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',

       'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',

       'SaleCondition', 'SalePrice']].corr(),annot = True, square = True, cmap = 'Blues',linewidth = 0.5)

print(cm)

plt.show()
#scatterplot

sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(df_train[cols], size = 2.5)

plt.show();
#bivariate analysis saleprice/grlivarea (Shows that how similar they are)

plt.subplot(2,1,1)

plt.scatter(df_train['SalePrice'],df_train['TotalBsmtSF'])

plt.title('Subplots of TotalBsmtSF Vs SalePrice and GrLivArea Vs SalePrice')

plt.ylabel('TotalBsmtSF')

plt.subplot(2,1,2)

plt.scatter(df_train['SalePrice'],df_train['GrLivArea'],color = 'g')

plt.xlabel('SalePrice')

plt.ylabel('GrLivArea')

plt.show()
plt.scatter(df_train['GrLivArea'], df_train['SalePrice'])

plt.title('Relation b/w Gr Living Area and the SalePrice')

plt.xlabel('GrLivArea')

plt.ylabel('SalePrice')

plt.show()