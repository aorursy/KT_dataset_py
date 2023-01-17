# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy import stats

from scipy.stats import norm, skew #for some statistics

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', 100)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')

test = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')
# Preview of the train dataset

train.head()
# Identify the number of numeric and non-numeric columns

print(train.select_dtypes(include='object').shape[1])

print()

print(train.select_dtypes(exclude='object').shape[1])
# Statistics summary of the train dataset

train.describe()
# The 23 nominal categorical variables

nominal_variables = ['MSSubClass', 'MSZoning', 'Street', 'Alley','LandContour',

                     'LotConfig', 'Neighborhood', 'Condition1','Condition2', 'BldgType',

                     'HouseStyle', 'RoofStyle', 'RoofMatl','Exterior1st', 'Exterior2nd',

                     'MasVnrType', 'Foundation', 'Heating', 'CentralAir', 'GarageType',

                     'MiscFeature', 'SaleType', 'SaleCondition']
# The 23 ordinal categorical variables

ordinal_variables = ['LotShape', 'Utilities', 'LandSlope', 'OverallQual', 'OverallCond',

                     'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure',

                     'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'Electrical', 'KitchenQual',

                     'Functional', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond',

                     'PoolQC', 'Fence', 'PavedDrive']
# The 20 continuous variables

continuous_variables = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',

                        'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',

                        'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',

                        '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'SalePrice']
# The 14 discrete variables

discrete_variables = ['YearBuilt', 'YearRemodAdd', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',

                      'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',

                      'GarageYrBlt', 'GarageCars', 'MoSold', 'YrSold']
# import visualization libraries

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

sns.set(style='darkgrid')
# Plot scatter plot of SalePrice vs. GrLivArea

plt.figure(figsize=(10,8))

sns.scatterplot(train['GrLivArea'], train['SalePrice'])

plt.axvline(x=4000, c='r', linewidth=2)

plt.title('Scatter Plot of Sale Price vs. Ground Living Area')

plt.show()
# We can now drop those rows

train = train.drop(train[train['GrLivArea'] > 4000].index)

print(train.shape)

print()

print(test.shape)
# highly correlated features

correlation = train.corr()

top_correlation = correlation.index[abs(correlation["SalePrice"])>0.5]

plt.figure(figsize=(10,10))

g = sns.heatmap(train[top_correlation].corr(),annot=True,cmap="coolwarm")
# Plot histogram and probability plot before log transform

plt.figure(figsize=(8,10))

plt.subplot(2,1,1)

sns.distplot(train['SalePrice'], bins=30)

plt.axvline(x=train['SalePrice'].mean(), c='k', linewidth=2)

plt.title('Histogram of Sale Prices')

plt.show()



plt.figure(figsize=(10,10))

plt.subplot(2,2,1)

stats.probplot(train['SalePrice'], plot=plt)

plt.title('Probability plot of Sale Prices')

plt.show()
# Check skewness of the target variable

train['SalePrice'].skew()
# Create a copy and perform log transform

train_copy1 = train.copy()

train_copy1['SalePrice'] = np.log(train_copy1['SalePrice'])
# Import scipy special's boxcox library

from scipy.special import boxcox1p



# we will not go into detail on which lambda to select but the idea is

# the lambda will affect the transformed data's skewness

train_copy2 = train.copy()

lam = 0.15

train_copy2['SalePrice'] = boxcox1p(train_copy2['SalePrice'], lam)
# Plot histogram and probability plots

plt.figure(figsize=(20, 10))

plt.subplot(2,2,1)

sns.distplot(train_copy1['SalePrice'], bins=30)

plt.axvline(train_copy1['SalePrice'].mean(), c='k', linewidth=2)

plt.title('Histogram of Log Transformed Sale Prices')



plt.figure(figsize=(20, 10))

plt.subplot(2,2,2)

sns.distplot(train_copy2['SalePrice'], bins=30)

plt.axvline(train_copy2['SalePrice'].mean(), c='k', linewidth=2)

plt.title('Histogram of Box-Cox Transformed Sale Prices')



plt.figure(figsize=(20,10))

plt.subplot(2,2,3)

stats.probplot(train_copy1['SalePrice'], plot=plt)

plt.title('Probability plot of Log Transformed Sale Prices')



plt.figure(figsize=(20,10))

plt.subplot(2,2,4)

stats.probplot(train_copy2['SalePrice'], plot=plt)

plt.title('Probability plot of Box-Cox Transformed Sale Prices')



plt.show()
# Skew values after transformation

log_skew = train_copy1['SalePrice'].skew()

bc_skew = train_copy2['SalePrice'].skew()



print('Log Transform: {:.3f}\nBox-Cox Transform: {:.3f}'.format(log_skew, bc_skew))
train['SalePrice'] = np.log(train['SalePrice'])
# look at the null values in train set

train.isnull().sum().sort_values(ascending=False)[:19]
# look at null values in test set

test.isnull().sum().sort_values(ascending=False)[:33]
y = train[(train['BsmtExposure'].isnull()) & (train['BsmtQual'].notnull())].index

train.loc[y, 'BsmtExposure'] = train.loc[y, 'BsmtExposure'].fillna('No')



x = train[(train['BsmtFinType2'].isnull()) & (train['BsmtQual'].notnull())].index

train.loc[x, 'BsmtFinType2'] = train.loc[x, 'BsmtFinType2'].fillna('Unf')
# label the columns with 'NA' as a category. With an exception for 'GarageYrBlt' as those with no GarageQual

# means there isnt a garage to begin with which will be filled with zeros.

col_to_fill_NA = ['PoolQC', 'MiscFeature', 'Alley', 'Fence',

                  'FireplaceQu', 'GarageCond', 'GarageType', 'GarageFinish',

                  'GarageQual', 'BsmtQual', 'BsmtCond', 'BsmtExposure',

                  'BsmtFinType2', 'BsmtFinType1', 'MasVnrType']



train[col_to_fill_NA] = train[col_to_fill_NA].fillna('NA')
train['GarageYrBlt'] = train['GarageYrBlt'].fillna(0)

train['MasVnrArea'] = train['MasVnrArea'].fillna(0)

train['Electrical'] = train['Electrical'].fillna('SBrkr')
train['LotFrontage'] = train.groupby('Neighborhood')['LotFrontage'].apply(lambda x: x.fillna(x.median()))
id1 = test[test['Id'] == 2127].index

id2 = test[test['Id'] == 2577].index
test.loc[id1,'GarageYrBlt'] = test.loc[id1, 'GarageYrBlt'].fillna(test.loc[id1,'YearBuilt'].values[0])

test.loc[id1,'GarageFinish'] = (test

                                .groupby('GarageType')['GarageFinish']

                                .apply(lambda x: x.fillna(x.mode().values[0]))

                               )

test.loc[id1,'GarageQual'] = (test

                              .groupby(['GarageYrBlt', 'GarageType'])['GarageQual']

                              .apply(lambda x: x.fillna(x.mode().values[0]))

                             )

test.loc[id1,'GarageCond'] = (test

                              .groupby(['GarageYrBlt', 'GarageType'])['GarageCond']

                              .apply(lambda x: x.fillna(x.mode().values[0]))

                             )
test.loc[id2,'GarageYrBlt'] = test.loc[id2, 'GarageYrBlt'].fillna(test.loc[id1,'YearBuilt'].values[0])

test.loc[id2,'GarageFinish'] = (test

                                .groupby('GarageType')['GarageFinish']

                                .apply(lambda x: x.fillna(x.mode().values[0]))

                               )

test.loc[id2,'GarageQual'] = (test

                              .groupby(['GarageYrBlt', 'GarageType'])['GarageQual']

                              .apply(lambda x: x.fillna(x.mode().values[0]))

                             )

test.loc[id2,'GarageCond'] = (test

                              .groupby(['GarageYrBlt', 'GarageType'])['GarageCond']

                              .apply(lambda x: x.fillna(x.mode().values[0]))

                             )

test.loc[id2, 'GarageCars'] = (test

                               .groupby(['GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond'])['GarageCars']

                               .apply(lambda x: x.fillna(x.median()))

                              )

test.loc[id2, 'GarageArea'] = (test

                               .groupby(['GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond'])['GarageArea']

                               .apply(lambda x: x.fillna(x.median()))

                              )
id3 = test[test['MasVnrArea'].notnull() & test['MasVnrType'].isnull()].index

test.loc[id3, 'MasVnrType'] = test.groupby('MasVnrArea')['MasVnrType'].apply(lambda x: x.fillna(x.mode().values[0]))
test.loc[:,['BsmtHalfBath', 'BsmtFullBath']] = test.loc[:,['BsmtHalfBath', 'BsmtFullBath']].fillna(0)

test[['Exterior1st', 'Exterior2nd']] = test[['Exterior1st', 'Exterior2nd']].fillna('VinylSd')

test['MSZoning'] = test['MSZoning'].fillna('RL')

test['Utilities'] = test['Utilities'].fillna('AllPub')

test['Functional'] = test['Functional'].fillna('Typ')

test['SaleType'] = test['SaleType'].fillna('WD')

test['KitchenQual'] = test['KitchenQual'].fillna('TA')

test['MasVnrArea'] = test['MasVnrArea'].fillna(0)
id4 = test[test['Id'] == 2121].index

test.loc[id4, ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']] = test.loc[id4, ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']].fillna(0)
# locate the ids with either BsmtQual is missing but with BsmtCond values and vice versa

id5 = test[(

            test['BsmtCond'].isnull() & test['BsmtQual'].notnull()

            ) |  (

                  test['BsmtCond'].notnull() & test['BsmtQual'].isnull()

                  )

          ].index



test.loc[id5, 'BsmtQual'] = test.loc[id5, 'BsmtQual'].fillna('TA')

test.loc[id5, 'BsmtCond'] = test.loc[id5, 'BsmtQual'].fillna('TA')
id6 = test[(test['BsmtExposure'].isnull()) & (test['BsmtQual'].notnull())].index

test.loc[id6, 'BsmtExposure'] = test.loc[id6, 'BsmtExposure'].fillna('No')
test[col_to_fill_NA] = test[col_to_fill_NA].fillna('NA')
test['GarageYrBlt'] = test['GarageYrBlt'].fillna(0)
test['LotFrontage'] = test.groupby('Neighborhood')['LotFrontage'].apply(lambda x: x.fillna(x.median()))
# Double checking the datasets

print(train.isnull().sum())

print()

print(test.isnull().sum())
# Combine both data sets and drop the 'Id' column

df = train.append(test)

df = df.drop('Id', axis=1)
# Convert these variables to str type

df[['MSSubClass', 'OverallQual', 'OverallCond']] = df[['MSSubClass', 'OverallQual', 'OverallCond']].astype(str)

df[['MoSold', 'YrSold']] = df[['MoSold', 'YrSold']].astype(str)
# Creating new features and lowering the cardinality of the dataset

df['TotalHouseSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

df['TotalPorchSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']

df['TotalBathrooms'] = df['BsmtFullBath'] + df['BsmtHalfBath'] + df['FullBath'] + df['HalfBath']

df['HasPool'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

df['HasGarage'] = df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

df['HasBasement'] = df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

df['HasFireplace'] = df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

df['HasWoodDeck'] = df['WoodDeckSF'].apply(lambda x: 1 if x > 0 else 0)
# Drop the unwanted columns

drop_col = ['1stFlrSF', '2ndFlrSF', 'BsmtFullBath',

            'BsmtHalfBath', 'FullBath', 'HalfBath', 'OpenPorchSF',

            'EnclosedPorch', '3SsnPorch', 'ScreenPorch']

df = df.drop(drop_col, axis=1)
# Create a new dataframe consist of only numeric variables

num_df = df.select_dtypes(exclude = 'object')



# Check skew of all numerical features

num_skew = num_df.apply(lambda x: x.skew()).sort_values(ascending=False)

skew_df = pd.DataFrame({'Skew': num_skew})

skew_df
high_skew_df = skew_df[(skew_df['Skew']>0.5) | (skew_df['Skew']<-0.5)]



# Exclude new features and year columns

exclude_features = ['HasPool', 'HasGarage', 'HasBasement', 'HasFireplace', 'HasWoodDeck', 'YearBuilt', 'GarageYrBlt']

high_skew_df = high_skew_df[high_skew_df.index.isin(exclude_features) == False]

high_skew_features = high_skew_df.index



# Perform box-cox transformation with specified lambda

lam = 0.15

for feat in high_skew_features:

    df[feat] = boxcox1p(df[feat], lam)
# perform label encoding

from sklearn.preprocessing import LabelEncoder

    

for col in ordinal_variables:

    df[col] = LabelEncoder().fit_transform(df[col])
df.head()
# Perform one hot encoding

from sklearn.preprocessing import OneHotEncoder



nominal_variables = ['MSSubClass', 'MSZoning', 'Street', 'Alley','LandContour',

                     'LotConfig', 'Neighborhood', 'Condition1','Condition2', 'BldgType',

                     'HouseStyle', 'RoofStyle', 'RoofMatl','Exterior1st', 'Exterior2nd',

                     'MasVnrType', 'Foundation', 'Heating', 'CentralAir', 'GarageType',

                     'MiscFeature', 'SaleType', 'SaleCondition', 'MoSold', 'YrSold']



encoded_features = []

for col in nominal_variables:

    encoded_feat = OneHotEncoder().fit_transform(df[col].values.reshape(-1, 1)).toarray()

    n = df[col].nunique()

    cols = ['{}_{}'.format(col, n) for n in range(1, n + 1)]

    encoded_df = pd.DataFrame(encoded_feat, columns=cols)

    encoded_df.index = df.index

    encoded_features.append(encoded_df)



df = pd.concat([df, *encoded_features], axis=1).drop(nominal_variables, axis=1)



df.head()
# Now we can split the data into train and test sets again

df_train = df[:1456]

df_test = df[1456:]
# Compute training and test variables

X_train = df_train.drop('SalePrice', axis=1)

y_train = df_train['SalePrice']

X_test = df_test.drop('SalePrice', axis=1)
# Just to confirm that both datasets has the same amount of columns

print(X_train.shape)

print(X_test.shape)
# import necessary libraries

from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.model_selection import cross_val_score, KFold, train_test_split
# Create variables for all import regression models

lin = LinearRegression()

ridge = Ridge()

lasso = Lasso()

rf = RandomForestRegressor()

gb = GradientBoostingRegressor()
# Define a function to calculate rsme for different models



kfold = KFold(n_splits=10)

def rmsle_cv(model):

    kfold = KFold(n_splits=10)

    rmse= np.sqrt(abs(cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = kfold)))

    return(rmse)
# Linear Regression

score = rmsle_cv(lin)

print('Linear Regression score: {:.3f}'.format(score.mean()))
# Ridge Regression

score = rmsle_cv(ridge)

print('Ridge Regression score: {:.3f}'.format(score.mean()))
# Lasso Regression

score = rmsle_cv(lasso)

print('Lasso Regression score: {:.3f}'.format(score.mean()))
# Gradient Boosting

score = rmsle_cv(gb)

print('Gradient Boosting score: {:.3f}'.format(score.mean()))
# Random Forest

score = rmsle_cv(rf)

print('Random Forest score: {:.3f}'.format(score.mean()))
# Fitting the ridge model

ridge_model = ridge.fit(X_train, y_train)



# Predicting prices

X_pred = ridge_model.predict(X_test)
X_pred = np.expm1(X_pred)
# Compute submission dataframe



output = pd.DataFrame()

output['Id'] = test['Id']

output['SalePrice'] = X_pred

output.to_csv('submission.csv',index=False)