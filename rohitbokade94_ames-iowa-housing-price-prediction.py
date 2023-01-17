# Importing the libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline
# Reading in the train file

df_train = pd.read_csv("../input/train.csv")
# Column Names

df_train.columns
df_train.head()
df_train.describe()
dtypes = df_train.dtypes.reset_index()
# Missing data exploration

missing = df_train.isnull().sum().sort_values(ascending = False)

missing = missing.reset_index()

missing['Percent'] = missing.iloc[:, 1].apply(lambda x: x*100/np.sum(missing.iloc[:, 1]))

missing.columns = ['Attributes', 'Missing', 'Percent']

gtz = missing['Missing'] > 0

missing = missing[gtz]

missing
sns.barplot(x = 'Percent', y = 'Attributes', data = missing, palette = 'viridis')
df_train['PoolQC'].value_counts()

df_train['PoolQC'].fillna('No Pool', inplace = True)
# MiscFeature: 

df_train['MiscFeature'].fillna('None', inplace = True)

df_train['MiscFeature'].value_counts()
df_train['Alley'].fillna('No alley access', inplace = True)

df_train['Alley'].value_counts()
df_train['Fence'].fillna('No Fence', inplace = True)

df_train['Fence'].value_counts()
df_train['FireplaceQu'].fillna('No Fireplace', inplace = True)

df_train['FireplaceQu'].value_counts()
df_train['GarageType'].fillna('No Garage', inplace = True)

df_train['GarageType'].value_counts()
df_train['GarageCond'].fillna('No Garage', inplace = True)

df_train['GarageCond'].value_counts()
df_train['GarageFinish'].fillna('No Garage', inplace = True)

df_train['GarageFinish'].value_counts()
df_train['GarageQual'].fillna('No Garage', inplace = True)

df_train['GarageQual'].value_counts()
df_train['BsmtFinType2'].fillna('No Basement', inplace = True)

df_train['BsmtFinType2'].value_counts()
df_train['BsmtExposure'].fillna('No Basement', inplace = True)

df_train['BsmtExposure'].value_counts()
df_train['BsmtQual'].fillna('No Basement', inplace = True)

df_train['BsmtQual'].value_counts()
df_train['BsmtCond'].fillna('No Basement', inplace = True)

df_train['BsmtCond'].value_counts()
df_train['BsmtFinType1'].fillna('No Basement', inplace = True)

df_train['BsmtFinType1'].value_counts()
df_train['MasVnrType'].fillna('None', inplace = True)

df_train['MasVnrType'].value_counts()
df_train['Electrical'].value_counts()

df_train['Electrical'].fillna(value = 'SBrkr', inplace = True) # Filling the NaN values with mode 'SBrkr'

df_train['Electrical'].value_counts()
df_train['MasVnrArea'].fillna(0, inplace = True) # Filling the missing values with 0 as they do not have Mason Veneer
df_train['GarageYrBlt'].fillna('No Garage', inplace = True)
df_train['LotFrontage'].fillna(0, inplace = True)
# Converting into Categorical Variables

cat_vars = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',

       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'RoofStyle',

       'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',

       'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',

       'Functional', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond', 'YrSold', 'MoSold', 

       'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition', 'YearBuilt', 'YearRemodAdd']

for col in cat_vars:

    df_train[col] = df_train[col].astype('category',copy=False)
# Creating Dummy Variables

cat = pd.get_dummies(df_train[cat_vars], drop_first = True)

df = df_train

df = pd.concat([df, cat], axis = 1)

df.drop(cat_vars, axis = 1, inplace = True)
# Loading sklearn packages

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
X = df.drop(['SalePrice'], axis = 1).values

y = np.log10(df['SalePrice'].values)
# Splitting the dataset into train ans test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
# Instantiating Linear Regression Model

reg_all = LinearRegression()
# 5-Fold Cross-Validation

from sklearn.cross_validation import cross_val_score

cv_scores = cross_val_score(reg_all, X_train, y_train, cv = 5)
# Print the 5-fold cross-validation scores

print(cv_scores)

print("Average 5-Fold CV Score: {}".format(cv_scores.mean()))
# Fitting a Linear Regression Model

reg_all.fit(X_train, y_train)
# Predicting on training set

train_y_pred = reg_all.predict(X_train)
# Evaluating Performance Measures

print("R^2: {}".format(reg_all.score(X_train, y_train)))

rmse = np.sqrt(mean_squared_error(y_train, train_y_pred))

print("Root Mean Squared Error: {}".format(rmse))
# Predictig on test set

y_pred = reg_all.predict(X_test)
# Evaluating Performance Measures

print("R^2: {}".format(reg_all.score(X_test, y_test)))

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Root Mean Squared Error: {}".format(rmse))
# Creating a Histogram of Residuals

sns.distplot(y_test - y_pred)

plt.title('Distribution of residuals')
# Scatterplot of Predictions Vs. Actual Values

sns.regplot(y = train_y_pred, x = y_train, color = 'blue', label = 'Training Data', scatter_kws={'alpha':0.75})

sns.regplot(y = y_pred, x = y_test, color = 'green', label = 'Validation Data', scatter_kws={'alpha':0.75})

plt.title('Predicted Values vs Test Values')

plt.xlabel('Real Values')

plt.ylabel('Predicted Values')

plt.legend(loc = 'upper left')
# Plotting Residuals

sns.regplot(x = train_y_pred, y = train_y_pred - y_train, label = 'Training Data', color = 'blue', scatter_kws={'alpha':0.75})

sns.regplot(x = y_pred, y = y_pred - y_test, label = 'Validation Data', color = 'green', scatter_kws={'alpha':0.75})

plt.legend(loc = 'best')

plt.title('Residuals plot')

plt.ylabel('Residuals')

plt.xlabel('Predicted Values')
# Correlation plot for all the numeric variables in the dataset

corrmat = df_train.corr()

f, ax = plt.subplots(figsize=(15, 12))

_ = sns.heatmap(corrmat, linecolor = 'white', cmap = 'magma', linewidths = 3)
# Highly correlated variables

correlations = df_train.corr()

correlations = correlations.iloc[:36, :36] 

cut_off = 0.5

high_corrs = correlations[correlations.abs() > cut_off][correlations.abs() != 1].unstack().dropna().to_dict()

high_corrs = pd.Series(high_corrs, index = high_corrs.keys())

high_corrs = high_corrs.reset_index()

high_corrs = pd.DataFrame(high_corrs)

high_corrs.columns = ['Attributes', 'Correlations']

high_corrs['Correlations'] = high_corrs['Correlations'].drop_duplicates(keep = 'first')

high_corrs.dropna().sort_values(by = 'Correlations', ascending = False)
# Correlation with the SalePrice variable

corr = df_train.corr()['SalePrice']

corr[np.argsort(corr, axis=0)[::-1]] 

# OverallQual seems to have the highest correlation with the 'SalePrice' which is quite intuitive
mvp_list = ['GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 

            'TotRmsAbvGrd']

num_vars = ['GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 

            'TotRmsAbvGrd', 'SalePrice']
# Variables highly correlated with SalePrice

k = 10

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df_train[cols].values.T)

sns.set(font_scale=1.25)

f, ax = plt.subplots(figsize=(12, 9))

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, linewidth = 5,

                 yticklabels=cols.values, xticklabels=cols.values, cmap = 'viridis', linecolor = 'white')

plt.show()
# Plotting the distribution of SalePrice variable

sns.set_style('whitegrid')

fig = plt.figure(figsize=(15,5))

ax1 = fig.add_subplot(211)

_ = sns.distplot(df_train['SalePrice'], hist=True, kde=True, bins=50)

ax2 = fig.add_subplot(212)

_ = sns.distplot(np.log10(df_train['SalePrice']), hist=True, kde=True, bins=50)

print('Skewness: %f' %  df_train['SalePrice'].skew())

print('Kurtosis: %f' %  df_train['SalePrice'].kurtosis())
plt.figure(figsize = (15, 5))

sns.boxplot('SalePrice', data = df_train, palette = 'viridis')
df_train[df_train['SalePrice'] > 600000][['SalePrice', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', 

                                          '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd']]
f = pd.melt(df_train, id_vars = 'SalePrice', value_vars = mvp_list)

g = sns.FacetGrid(f, col = "variable",  col_wrap=2, sharex=False, sharey=False, size=5)

g.map(sns.regplot, 'value', 'SalePrice')
f = pd.melt(df_train, id_vars = 'SalePrice', value_vars = cat_vars)

g = sns.FacetGrid(f, col = "variable",  col_wrap = 2, sharex = False, sharey = False, size = 10)

g.map(sns.boxplot, 'value', 'SalePrice', palette = 'viridis')
# Selected categorical variables

cat_mvp_vars = ['MSSubClass', 'MSZoning', 'Utilities', 'Neighborhood', 'Condition1', 'Condition2', 'HouseStyle', 

                'OverallQual', 'OverallCond', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'HeatingQC', 'CentralAir',

                'KitchenQual', 'GarageFinish', 'GarageQual', 'PoolQC', 'SaleType', 'YearRemodAdd']
# List of final features

features = mvp_list + cat_mvp_vars
# Creating Dummy Variables

cat = pd.get_dummies(df_train[cat_mvp_vars], drop_first = True)

df = df_train[num_vars]

df = pd.concat([df, cat], axis = 1)
X = df.drop(['SalePrice'], axis = 1).values

y = np.log10(df['SalePrice'].values)
# Splitting the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
# Instantiate a Linear Regression Model 

reg_all = LinearRegression()
# Fitting a Linear Regression Model

reg_all.fit(X_train, y_train)

reg_all_coef = reg_all.fit(X_train, y_train).coef_
# 5- Fold Cross Validation

cv_scores = cross_val_score(reg_all, X_train, y_train, cv = 10)

# Print the 5-fold cross-validation scores

print(cv_scores)

print("Average 5-Fold CV Score: {}".format(cv_scores.mean()))
# Predicting on the training set

train_y_pred = reg_all.predict(X_train)
# Evaluating Performance Measures on training set

print("R^2: {}".format(reg_all.score(X_train, y_train)))

rmse = np.sqrt(mean_squared_error(y_train, train_y_pred))

print("Root Mean Squared Error: {}".format(rmse))
# Creating a Histogram of Residuals

sns.distplot(y_train - train_y_pred)

plt.title('Distribution of residuals')
# Predictig on test set

y_pred = reg_all.predict(X_test)
# Evaluating Performance Measures on validation set

print("R^2: {}".format(reg_all.score(X_test, y_test)))

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Root Mean Squared Error: {}".format(rmse))
%matplotlib inline

# Creating a Histogram of Residuals

sns.distplot(y_test - y_pred)

plt.title('Distribution of residuals')
# Scatterplot of Predictions Vs. Actual Values

sns.regplot(y = train_y_pred, x = y_train, color = 'blue', label = 'Training Data', scatter_kws={'alpha':0.75})

sns.regplot(y = y_pred, x = y_test, color = 'green', label = 'Validation Data', scatter_kws={'alpha':0.75})

plt.title('Predicted Values vs Test Values')

plt.xlabel('Real Values')

plt.ylabel('Predicted Values')

plt.legend(loc = 'upper left')
# Plotting Residuals

sns.regplot(x = train_y_pred, y = train_y_pred - y_train, label = 'Training Data', color = 'blue', scatter_kws={'alpha':0.75})

sns.regplot(x = y_pred, y = y_pred - y_test, label = 'Validation Data', color = 'green', scatter_kws={'alpha':0.75})

plt.legend(loc = 'best')

plt.title('Residuals plot')

plt.ylabel('Residuals')

plt.xlabel('Predicted Values')