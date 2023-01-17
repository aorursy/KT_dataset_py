import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train.head(5)

train.tail(5)
train.describe(include= 'all')
train.info()
train.shape
train.columns
numeric_columns = train.select_dtypes(include='number')
numeric_columns.shape
numeric_columns.isnull().sum()
median1 = train['LotFrontage'].median()
median2 = train['GarageYrBlt'].median()
train['LotFrontage'].replace(np.nan, median1, inplace = True)
train['GarageYrBlt'].replace(np.nan, median2, inplace = True)
numeric_columns.isnull().sum()
Category_columns = train.select_dtypes(include='object')
Category_columns.shape
Category_columns.isnull().sum()
train['Alley'].fillna(train['Alley'].mode()[0], inplace=True)
train['BsmtQual'].fillna(train['BsmtQual'].mode()[0], inplace=True)
train['BsmtCond'].fillna(train['BsmtCond'].mode()[0], inplace=True)
train['BsmtExposure'].fillna(train['BsmtExposure'].mode()[0], inplace=True)
train['BsmtFinType1'].fillna(train['BsmtFinType1'].mode()[0], inplace=True)
train['BsmtFinType2'].fillna(train['BsmtFinType2'].mode()[0], inplace=True)
train['FireplaceQu'].fillna(train['FireplaceQu'].mode()[0], inplace=True)
train['GarageType'].fillna(train['GarageType'].mode()[0], inplace=True)
train['GarageFinish'].fillna(train['GarageFinish'].mode()[0], inplace=True)
train['GarageQual'].fillna(train['GarageQual'].mode()[0], inplace=True)
train['GarageCond'].fillna(train['GarageCond'].mode()[0], inplace=True)
train['PoolQC'].fillna(train['PoolQC'].mode()[0], inplace=True)
train['Fence'].fillna(train['Fence'].mode()[0], inplace=True)
train['MiscFeature'].fillna(train['MiscFeature'].mode()[0], inplace=True)
train['Electrical'].fillna(train['Electrical'].mode()[0], inplace=True)
Category_columns.isnull().sum()
train.isnull().sum()
duplicate = train.duplicated()
print(duplicate.sum())
train[duplicate]
train.drop_duplicates(inplace = True)
sns.heatmap(train.corr(), fmt = ".2f")
numeric_features = numeric_columns.columns
numeric_features
year_feature = [feature for feature in numeric_features if 'Yr' in feature or 'Year' in feature]

year_feature
## Lets analyze the Temporal Datetime Variables
## We will check whether there is a relation between year the house is sold and the sales price
train.groupby('YrSold')['SalePrice'].median().plot()
plt.xlabel('Year Sold')
plt.ylabel('Median House Price')
plt.title("House Price vs YearSold")
for feature in year_feature:
    if feature!='YrSold':
        data=train.copy()
        ## We will capture the difference between year variable and year the house was sold for
        data[feature]=data['YrSold']-data[feature]

        plt.scatter(data[feature],data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.show()
discrete_feature=[feature for feature in numeric_features if len(train[feature].unique())<25 and feature not in year_feature+['Id']]
print("Discrete Variables Count: {}".format(len(discrete_feature)))
discrete_feature
for feature in discrete_feature:
    data=train.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()
continuous_feature=[feature for feature in numeric_features if feature not in discrete_feature+year_feature+['Id']]
print("Continuous feature Count {}".format(len(continuous_feature)))
continuous_feature

for feature in continuous_feature:
    data=train.copy()
    data[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title(feature)
    plt.show()
train['MasVnrArea'].unique()

for feature in continuous_feature:
    data=train.copy()
    if 0 in data[feature].unique():
        pass
    else:
        plt.scatter(data[feature],data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalesPrice')
        plt.title(feature)
        plt.show()

for feature in continuous_feature:
    data=train.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature]=np.log(data[feature])
        data.boxplot(column=feature)
        plt.ylabel(feature)
        plt.title(feature)
        plt.show()
categorical_features=[feature for feature in train.columns if data[feature].dtypes=='O']
categorical_features
for feature in categorical_features:
    data=train.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()
