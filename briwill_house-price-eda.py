import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# This prevents the middle columns on wide dataframes from being "abbreviated" (...) in the display
pd.options.display.max_columns = None

# Load the train datafrme
train_df = pd.read_csv('../input/train.csv')

# Take a look at the first 5 columns
train_df.head()
# What kinds of values are in "Fence" and "MiscFeature"? Include null values
print ("Fence values and counts: \n", train_df['Fence'].value_counts(dropna=False))
print ("MiscFeature values and counts: \n", train_df['MiscFeature'].value_counts(dropna=False))
print ("MasVnrType values and counts: \n", train_df['MasVnrType'].value_counts(dropna=False))
# Put column names into lists by variable type, continuous or categorical

cat_cols = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 
            'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 
            'OverallQual', 'OverallCond', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 
            'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 
            'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 
            'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 
            'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleCondition', 'SaleType', 
            'MoSold']

cont_cols = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'MasVnrArea', 'BsmtFinSF1', 
             'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 
             'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 
             'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 
             'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 
             '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']

date_cols = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold']
# Print out the values in the continous columns to see if there are any
# visibly odd entries, like typos...I didn't see any problems
# Observations:
# - All categoricals will need to be label-encoded, as they are not numeric
# - CentralAir Y/N should be mapped to 1/0
# - Should create a HasPool column that is 1 if PoolQC is not null
# - Null category values should be their own category, i.e. like 0 or 'None'
# - Quality and Condition categories should be mapped: NaN, 'Po', 'Fa', 'TA', Gd', and 'Ex' 
#   to 0, 1, 2, 3, 4, and 5. 'TA' apparently stands for 'Typical/Average'

for col in cat_cols:
    print ("{} values and counts: \n".format(col), train_df[col].value_counts(dropna=False), "\n")
# Which categorical columns contain any nulls?
null_counts = train_df[cat_cols].isnull().sum()
null_counts[null_counts > 0].sort_values(ascending=False)
# Which continuous columns contain any nulls?
null_counts = train_df[cont_cols].isnull().sum()
null_counts[null_counts > 0].sort_values(ascending=False)
# Let's look at the Garage data. The counts of missing categorical and continuous values match,
# so they may be the same rows and could be for homes that don't have garages. Let's verify.
garage_cols = [col for col in train_df if 'Garage' in col]
train_df[train_df['GarageCars']==0][garage_cols]
# Let's also look at the Basement data. The counts of missing categorical and continuous values match,
# so they may be the same rows and could be for homes that don't have basements. Let's verify.
garage_cols = [col for col in train_df if 'Bsmt' in col]
print ("Shape: ", train_df[train_df['BsmtExposure'].isnull()][garage_cols].shape)
train_df[train_df['BsmtExposure'].isnull()][garage_cols]
# There's only one house (index 948) that has a basement with a missing BsmtExposure that has a 
# basement. I'll fill in the missing value with the "mode"  (most frequently occuring condition)
mode_val = train_df['BsmtExposure'].mode()[0]
train_df.loc[(train_df['BsmtExposure'].isnull()) & (train_df['TotalBsmtSF'] != 0), 'BsmtExposure'] = mode_val

# It looks like there also must be a home with a null BsmtFinType2 that don't appear in the
# list above. Let's examine to make sure there's nothing odd going on.
print ("Shape: ", train_df[train_df['BsmtFinType2'].isnull()][garage_cols].shape)
train_df[train_df['BsmtFinType2'].isnull()][garage_cols]
# There's only one house (index 332) that has a basement with a missing BsmtFinType2.
# I'll fill in that missing value with the "mode" condition (most frequently occuring condition)
mode_val = train_df['BsmtFinType2'].mode()[0]
train_df.loc[(train_df['BsmtFinType2'].isnull()) & (train_df['TotalBsmtSF'] != 0), 'BsmtFinType2'] = mode_val
# Let's clean up a few other missing values
train_df['MasVnrType'].fillna('None', inplace=True)
train_df['MasVnrArea'].fillna(0, inplace=True)
train_df['Electrical'].fillna(train_df['Electrical'].mode()[0], inplace=True)       
# To fill in missing LotFrontage info, I'll compute the ratio of LotFrontage to square
# root of LotSize, and then use the average LotRatio by LotShape to compute the missing
# LotFrontages.

# Compute LotRatios
train_df['LotRatio'] = train_df['LotFrontage'] / train_df['LotArea']**0.5
# Fill in missing LotRatios with mean values per LotShape category
train_df['LotRatio'] = train_df.groupby('LotShape')['LotRatio'].transform(lambda x: x.fillna(x.mean()))
# Fill in missing LotFrontage values
train_df['LotFrontage'].fillna(train_df['LotRatio'] * train_df['LotArea']**0.5, inplace=True)
# Drop LotRatio since we're done with it
train_df.drop(['LotRatio'], inplace=True, axis=1)
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
%matplotlib inline

# Let's use box plots to look for outlier values
box1 = sns.boxplot(data=train_df['SalePrice'], orient="h", palette="Set2")
box1.set_title('SalePrice')
plt.show()
box2 = sns.boxplot(data=train_df['LotArea'], orient="h", palette="Set2")
box2.set_title('LotArea')
plt.show()
box2 = sns.boxplot(data=train_df['GrLivArea'], orient="h", palette="Set2")
box2.set_title('GrLivArea')
plt.show()
cols = ['FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'GarageCars',]

box1 = sns.boxplot(data=train_df[cols], orient="h", palette="Set2")
# Visually, it looks like the outliers are houses with prices over $650k,
# with LotAreas over 100k square feet, above grade living areas over 4000 square feet,
# and total rooms above grade over 12. There's a fair amount of overlap here.

train_df[(train_df['SalePrice'] > 650000) | (train_df['LotArea'] > 100000) 
         | (train_df['GrLivArea'] > 4000) | (train_df['TotRmsAbvGrd'] > 12)]
cols = ['LotArea', 'OverallQual', 'OverallCond', 'GrLivArea']

g = sns.pairplot(train_df, y_vars=['SalePrice'], x_vars=cols)
cols = ['FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'GarageCars']

g = sns.pairplot(train_df, y_vars=['SalePrice'], x_vars=cols)
pair_cols = ['LotArea', 'OverallQual', 'OverallCond', 'GrLivArea', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'GarageCars', 'SalePrice']

plt.figure(figsize=(18, 14))
sns.heatmap(train_df[pair_cols].corr(), cmap="YlGnBu", annot=True, fmt='03.2f')
