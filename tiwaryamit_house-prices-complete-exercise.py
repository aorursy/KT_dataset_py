# Start time for script

import time

start = time.time() # mark start of running whole kernel



# pandas / os etc.

import pandas as pd

from pandas import Series, DataFrame

import scipy.stats as ss

import statsmodels.api as sm



# numpy, matplotlib, seaborn

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

import seaborn as sns



# To plot pretty figures

%matplotlib inline

plt.rcParams['axes.labelsize'] = 14

plt.rcParams['xtick.labelsize'] = 12

plt.rcParams['ytick.labelsize'] = 12

sns.set_style('dark', {'axes.facecolor' : 'lightgray'})



# for seaborn issue:

import warnings

warnings.filterwarnings('ignore')



# machine learning [Regression]

from sklearn.model_selection import (train_test_split, cross_val_score, cross_val_predict, KFold, GridSearchCV, StratifiedKFold)

from sklearn.preprocessing import (MaxAbsScaler, PolynomialFeatures)

from sklearn import (preprocessing, clone)

from sklearn.metrics import mean_squared_error



from sklearn.linear_model import (BayesianRidge, LinearRegression, Lasso, Ridge, ElasticNet, OrthogonalMatchingPursuit)

from sklearn.svm import SVR, LinearSVR

from sklearn.ensemble import (RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor, BaggingRegressor)

from sklearn.neighbors import KNeighborsRegressor

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeRegressor

from sklearn.neural_network import MLPRegressor

from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.gaussian_process.kernels import RBF

from xgboost import XGBRegressor

# XGBoost installation instruction - Windows [simple] - https://www.ibm.com/developerworks/community/blogs/jfp/entry/Installing_XGBoost_For_Anaconda_on_Windows?lang=en



# to make this notebook's output stable across runs

np.random.seed(42)
# Manual method

train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')

comb_data = pd.concat([train_data, test_data])

comb_data.head()
# Find basic stats of number (continuous and ordinal) features

train_data.describe().transpose()
# Find data types and no. of data in each feature

train_data.info()

print("----------------------------------------------------")

test_data.info()

print("----------------------------------------------------")

comb_data.info()
# Find columns with missing data

print("-------------Count of null values in Train Data------------------")

print(train_data.loc[:, train_data.isnull().any()].isnull().sum())

print("-------------Count of null values in Test Data------------------")

print(test_data.loc[:, test_data.isnull().any()].isnull().sum())
# Fill with median values [only columns in which missing values are very few]. There could be better way to fill columns with large no. of missing data.

for column_name in ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtUnfSF', 'GarageArea', 'GarageCars', 'MasVnrArea', 'TotalBsmtSF',\

                   'MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'Electrical', 'MasVnrType', 'SaleType', 'Functional', 'KitchenQual']:

    comb_data[column_name].fillna(comb_data[column_name].value_counts().index[0], inplace=True)
print("-------------Count of null values in Comb Data------------------")

print(comb_data.loc[:, comb_data.isnull().any()].isnull().sum())
# After reading data dictionary, I concluded that many columns with categorical data which means 'Not' [like 'No Basement'] instead of 'NA' were converted to NaN by Pandas.

# Fill null values which means 'Not Present'

for column_name in ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',\

                   'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu']:

    comb_data[column_name].fillna('Not', inplace=True)
# Find stats of remaining missing data

print(comb_data[['GarageYrBlt', 'LotFrontage']].describe().transpose())

print("-------------Count of null values in Comb Data------------------")

print(comb_data.loc[:, comb_data.isnull().any()].isnull().sum())
# Find correlation in other columns

print("-------------Count of null 'GarageYrBlt' and 'No Garage' in Comb Data------------------")

print(comb_data['GarageYrBlt'][(comb_data['GarageYrBlt'].isnull()) & (comb_data['GarageType'] == 'Not')].isnull().sum())
# Fill missing 'LotFrontage' with random numbers

comb_data.LotFrontage[np.isnan(comb_data.LotFrontage)] = np.random.normal(comb_data.LotFrontage.mean(), comb_data.LotFrontage.std(), np.isnan(comb_data.LotFrontage).sum())
# Fill null 'GarageYrBlt' values with 1900.

for column_name in ['GarageYrBlt']:

    comb_data[column_name].fillna(1900, inplace=True)

    

# Confirm - Find stats of remaining missing data

print(comb_data[['GarageYrBlt', 'LotFrontage']].describe().transpose())

print("-------------Count of null values in Comb Data------------------")

print(comb_data.loc[:, comb_data.isnull().any()].isnull().sum())
# Change categorical columns to category data type [By default, Pandas converted these columns to 'Object']

for column_name in comb_data.select_dtypes(include=['object']).columns:

    comb_data[column_name] = comb_data[column_name].astype('category')    

comb_data.dtypes
# Histograms and Scatter plots are quick way to visulalize numerical features

# Plot histograms for ordianl and continuos variables

comb_data.select_dtypes(include=[np.number]).hist(bins=50, figsize=(28, 38))

plt.show()
# Plot data and a linear regression model fit.

# Let us see how number columns look against 'SalePrice', which is target variable of this exercise.

f, axes = plt.subplots(10,4, figsize = (28, 60), sharey=True)

for i, col_name in enumerate(comb_data.select_dtypes(include=[np.number]).columns):

    row = i // 4

    col = i % 4

    ax_curr = axes[row, col]

    sns.regplot(x=col_name, y='SalePrice', data=comb_data, scatter=True, marker = '.', ax = ax_curr)

plt.show()
# Calculate correlation of "SalePrice" against other attributes (By default, it will take numerical columns only)

corr_matrix = comb_data.corr()

corr_matrix["SalePrice"].sort_values(ascending=False)
# As obeserved above, there are many features with ordinal categories. Let us try a better visualization for these features.

# Plot columns with ordinal categories columns with Mean 'SalePrice' on Y axis and count of observations in each category [Shown in box]

f, axes = plt.subplots(3,4, figsize = (28, 21), sharey=True)

for i, col_name in enumerate(comb_data.ix[:, comb_data.apply(lambda x: x.nunique()) < 20].select_dtypes(include=['int64']).columns):

    row = i // 4

    col = i % 4

    ax_curr = axes[row, col]    

    ax = sns.countplot(x=col_name, data=comb_data, ax = ax_curr)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2., height + 30000, '{:1.0f}'.format(height), ha="center", va="bottom", size=12, color='black', bbox=dict(facecolor='white', alpha=0.5), rotation=90)

    sns.barplot(col_name, 'SalePrice', data=comb_data, ax = ax_curr)

plt.show()
# Find if a column can be explained by other column, i.e. highly dependent (correlated)

# Drawing correlation matris - Standard Pearson coefficients

# Compute the correlation matrix

corr_mat = comb_data.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr_mat, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(28, 24))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(240, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr_mat, mask=mask, cmap=cmap, vmax=.9, center=0, square=True, annot=True, linecolor='black', linewidths=0, cbar_kws={"shrink": .4}, fmt='.2f')

plt.show()
# Drawing correlation matris - Kendall's Tau coefficient

# Compute the correlation matrix

corr_mat = comb_data.corr(method='kendall')



# Generate a mask for the upper triangle

mask = np.zeros_like(corr_mat, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(28, 24))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(240, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr_mat, mask=mask, cmap=cmap, vmax=.9, center=0, square=True, annot=True, linecolor='black', linewidths=0, cbar_kws={"shrink": .5}, fmt='.2f')

plt.show()
# Find categorical features that are being represented as ordinal feature

# Change categorical columns to category data type

for column_name in ['MSSubClass', 'YrSold']:

    comb_data[column_name] = comb_data[column_name].astype('category')
# Plot categorical variables Barplot with Mean 'SalePrice' on Y axis and count of observations in each category [Shown in box]

f, axes = plt.subplots(12,4, figsize = (28, 60), sharey=True)

for i, col_name in enumerate(comb_data.select_dtypes(include=['category']).columns):

    row = i // 4

    col = i % 4

    ax_curr = axes[row, col]    

    ax = sns.countplot(x=col_name, data=comb_data, ax = ax_curr)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2., height + 30000, '{:1.0f}'.format(height), ha="center", va="bottom", size=12, color='black', bbox=dict(facecolor='white', alpha=0.5), rotation=90)

    sns.barplot(col_name, 'SalePrice', data=comb_data, ax = ax_curr)

plt.show()
# Caluculate assciation between 2 categorical columns - Cramer's V score

for i in comb_data.select_dtypes(include=['category']).columns:

    col_1 = i

    for j in comb_data.select_dtypes(include=['category']).columns:

        col_2 = j

        if col_1 == col_2:

            break

        confusion_matrix = pd.crosstab(comb_data[col_1], comb_data[col_2])

        chi2 = ss.chi2_contingency(confusion_matrix)[0] # import scipy.stats as ss

        n = confusion_matrix.sum().sum()

        phi2 = chi2/n

        r,k = confusion_matrix.shape

        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    

        rcorr = r - ((r-1)**2)/(n-1)

        kcorr = k - ((k-1)**2)/(n-1)

        Cramer_V = np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

        if Cramer_V > 0.7:

            print("The Cramer's V score bettween " + col_1 + " and " + col_2 + " is : " + str(Cramer_V))

        result = Cramer_V
def outliers_iqr(df, columns_for_outliers):

    for column_name in columns_for_outliers:

        if not 'Outlier' in df.columns:

            df['Outlier'] = 0

        q_75, q_25 = np.percentile(df[column_name], [75 ,25])

        iqr = q_75 - q_25

        minm = q_25 - (iqr*1.75)

        maxm = q_75 + (iqr*1.75)        

        df['Outlier'] = np.where(df[column_name] > maxm, 1, np.where(df[column_name] < minm, 1, df['Outlier']))
# Drop rows with outlier data

columns_for_outliers = ['YearBuilt', 'GrLivArea', 'TotalBsmtSF', 'Fireplaces', 'GarageArea', 'LotArea']

outliers_iqr(comb_data, columns_for_outliers)

print('Total ' + str(comb_data.Outlier.sum()) + ' rows with outliers from comb_data were deleted')

comb_data = comb_data[comb_data.Outlier != 1]

comb_data = comb_data.drop(['Outlier'], axis=1)
# Drop rows with outlier data

# Delete more rows from 'LotFrontage' outliers

comb_data = comb_data[comb_data.LotFrontage < 130]
# Create new columns, MasVnr - Y or N, OpenPorch - Y or N

comb_data['MasVnr'] = np.where(comb_data['MasVnrArea'] == 0, 0, 1)

comb_data['OpenPorch'] = np.where(comb_data['OpenPorchSF'] == 0, 0, 1)



# Change to category data type

for column_name in ['MasVnr', 'OpenPorch']:

    comb_data[column_name] = comb_data[column_name].astype('category')



# Create another column of Total bathrooms

comb_data['TotalBath'] = pd.to_numeric(comb_data.BsmtFullBath + comb_data.BsmtHalfBath*0.5 + comb_data.FullBath + comb_data.HalfBath*0.5)
# Drop unnecessary columns, these columns won't be useful in analysis and prediction

comb_data = comb_data.drop(['3SsnPorch', 'BsmtFinSF2', 'BsmtHalfBath', 'EnclosedPorch', 'KitchenAbvGr', 'LowQualFinSF', 'MiscVal', 'PoolArea', 'ScreenPorch',\

'BsmtUnfSF', 'Id', 'MoSold', 'OverallCond', 'YearRemodAdd', 'YrSold', 'BedroomAbvGr', 'BsmtFullBath',  'GarageCars',\

'TotRmsAbvGrd', 'BsmtFinSF1', '1stFlrSF', '2ndFlrSF', 'Alley', 'BldgType', 'BsmtCond', 'BsmtFinType2', 'CentralAir', 'Condition1', 'Condition2',\

'Electrical', 'Fence', 'ExterCond', 'Functional', 'GarageCond', 'GarageQual', 'Heating', 'LotConfig', 'LandContour', 'LandSlope',\

'MiscFeature', 'PavedDrive', 'PoolQC', 'RoofMatl', 'RoofStyle', 'SaleType', 'Street', 'Utilities', 'Exterior2nd', 'HouseStyle',\

'MasVnrArea', 'OpenPorchSF', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath'], axis=1)
# let us see how remaining number columns look against 'SalePrice', which is target variable of this exercise.

f, axes = plt.subplots(2,4, figsize = (30, 15), sharey=True)

for i, col_name in enumerate(comb_data.select_dtypes(include=['float64']).columns):

    row = i // 4

    col = i % 4

    ax_curr = axes[row, col]

    sns.regplot(x=col_name, y='SalePrice', data=comb_data, scatter=True, marker = '.', ax = ax_curr)

plt.show()



f, axes = plt.subplots(2,4, figsize = (30, 15), sharey=True)

for i, col_name in enumerate(comb_data.select_dtypes(include=['int64']).columns):

    row = i // 4

    col = i % 4

    ax_curr = axes[row, col]

    sns.regplot(x=col_name, y='SalePrice', data=comb_data, scatter=True, marker = '.', ax = ax_curr)

plt.show()
# Let us try lmplot for better visualization of top features

plt.figure(figsize=(28, 30))

for col_name in ['OverallQual', 'TotalBath', 'Fireplaces']:

    lmp = sns.lmplot(data = comb_data, x = 'GrLivArea', y = 'SalePrice', fit_reg = False, hue = col_name, palette = sns.color_palette('Greens', comb_data[col_name].nunique()), size = 6,\

               aspect = 1.8, scatter_kws = {"s":80})

    lmp.set(ylim=(50000, 400000), xlim=(500, None))

plt.show()
# Flush categories with 0 values in memory after dropping row

for column_name in comb_data.select_dtypes(include=['category']).columns:

    comb_data[column_name] = comb_data[column_name].astype('object')

    

# Change categorical columns to category data type [By default, Pandas converted these columns to 'Object']

for column_name in comb_data.select_dtypes(include=['object']).columns:

    comb_data[column_name] = comb_data[column_name].astype('category') 
# Plot categorical variables Barplot with Mean 'SalePrice' on Y axis and count of observations in each category [Shown in box]

f, axes = plt.subplots(5,4, figsize = (28, 35), sharey=True)

for i, col_name in enumerate(comb_data.select_dtypes(include=['category']).columns):

    row = i // 4

    col = i % 4

    ax_curr = axes[row, col]    

    ax = sns.countplot(x=col_name, data=comb_data, ax = ax_curr)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2., height + 30000, '{:1.0f}'.format(height), ha="center", va="bottom", size=12, color='black', bbox=dict(facecolor='white', alpha=0.5), rotation=90)

    sns.barplot(col_name, 'SalePrice', data=comb_data, ax = ax_curr)

plt.show()
# Caluculate assciation between 2 columns - Cramer's V score

for i in comb_data.select_dtypes(include=['category']).columns:

    col_1 = i

    for j in comb_data.select_dtypes(include=['category']).columns:

        col_2 = j

        if col_1 == col_2:

            break

        confusion_matrix = pd.crosstab(comb_data[col_1], comb_data[col_2])

        chi2 = ss.chi2_contingency(confusion_matrix)[0] # import scipy.stats as ss

        n = confusion_matrix.sum().sum()

        phi2 = chi2/n

        r,k = confusion_matrix.shape

        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))

        rcorr = r - ((r-1)**2)/(n-1)

        kcorr = k - ((k-1)**2)/(n-1)

        Cramer_V = np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

        if Cramer_V > 0.7:

            print("The Cramer's V score bettween " + col_1 + " and " + col_2 + " is : " + str(Cramer_V))

        result = Cramer_V
# Drop more columns as per final analysis

comb_data = comb_data.drop(['MasVnr'], axis=1)
#### Create DataFrames, Process Data, Transform Data for ML models
# Create independent variables dataframe

X_train = train_data

X_test = test_data
# Create new columns, OpenPorch - Y or N

X_train['OpenPorch'] = np.where(X_train['OpenPorchSF'] == 0, 0, 1)



X_test['OpenPorch'] = np.where(X_test['OpenPorchSF'] == 0, 0, 1)



# Create another column of Total bathrooms

X_train['TotalBath'] = pd.to_numeric(X_train['BsmtFullBath'] + X_train['BsmtHalfBath']*0.5 + X_train['FullBath'] + X_train['HalfBath']*0.5)



X_test['TotalBath'] = pd.to_numeric(X_test['BsmtFullBath'] + X_test['BsmtHalfBath']*0.5 + X_test['FullBath'] + X_test['HalfBath']*0.5)
# Drop unnecessary columns, these columns won't be useful in analysis and prediction

X_train = X_train.drop(['3SsnPorch', 'BsmtFinSF2', 'BsmtHalfBath', 'EnclosedPorch', 'KitchenAbvGr', 'LowQualFinSF', 'MiscVal', 'PoolArea', 'ScreenPorch',\

'BsmtUnfSF', 'Id', 'MoSold', 'OverallCond', 'YearRemodAdd', 'YrSold', 'BedroomAbvGr', 'BsmtFullBath',  'GarageCars', \

'TotRmsAbvGrd', 'BsmtFinSF1', '1stFlrSF', '2ndFlrSF', 'Alley', 'BldgType', 'BsmtCond', 'BsmtFinType2', 'CentralAir', 'Condition1', 'Condition2',\

'Electrical', 'Fence', 'ExterCond', 'Functional', 'GarageCond', 'GarageQual', 'Heating', 'LotConfig', 'LandContour', 'LandSlope',\

'MiscFeature', 'PavedDrive', 'PoolQC', 'RoofMatl', 'RoofStyle', 'SaleType', 'Street', 'Utilities', 'Exterior2nd', 'HouseStyle',\

'MasVnrArea', 'OpenPorchSF', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath'], axis=1)



X_test = X_test.drop(['3SsnPorch', 'BsmtFinSF2', 'BsmtHalfBath', 'EnclosedPorch', 'KitchenAbvGr', 'LowQualFinSF', 'MiscVal', 'PoolArea', 'ScreenPorch',\

'BsmtUnfSF', 'Id', 'MoSold', 'OverallCond', 'YearRemodAdd', 'YrSold', 'BedroomAbvGr', 'BsmtFullBath',  'GarageCars',\

'TotRmsAbvGrd', 'BsmtFinSF1', '1stFlrSF', '2ndFlrSF', 'Alley', 'BldgType', 'BsmtCond', 'BsmtFinType2', 'CentralAir', 'Condition1', 'Condition2',\

'Electrical', 'Fence', 'ExterCond', 'Functional', 'GarageCond', 'GarageQual', 'Heating', 'LotConfig', 'LandContour', 'LandSlope',\

'MiscFeature', 'PavedDrive', 'PoolQC', 'RoofMatl', 'RoofStyle', 'SaleType', 'Street', 'Utilities', 'Exterior2nd', 'HouseStyle',\

'MasVnrArea', 'OpenPorchSF', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath'], axis=1)
# 'Not' for 'NA'

for column_name in ['BsmtQual', 'BsmtExposure', 'BsmtFinType1', 'FireplaceQu', 'GarageType', 'GarageFinish']:

    X_train[column_name].fillna('Not', inplace=True)

    

for column_name in ['BsmtQual', 'BsmtExposure', 'BsmtFinType1', 'FireplaceQu', 'GarageType', 'GarageFinish']:

    X_test[column_name].fillna('Not', inplace=True)

    

# With most frequent values values

for column_name in ['GarageArea', 'TotalBsmtSF', 'MSZoning', 'Exterior1st', 'MasVnrType', 'KitchenQual', 'TotalBath']:

    X_train[column_name].fillna(X_train[column_name].value_counts().index[0], inplace=True)

    

for column_name in ['GarageArea', 'TotalBsmtSF', 'MSZoning', 'Exterior1st', 'MasVnrType', 'KitchenQual', 'TotalBath']:

    X_test[column_name].fillna(X_test[column_name].value_counts().index[0], inplace=True)

    

# Find stats of remaining missing data

print(X_train[['GarageYrBlt', 'LotFrontage']].describe().transpose())

print("-------------Count of null values in X_train Data------------------")

print(X_train.loc[:, X_train.isnull().any()].isnull().sum())



# Find stats of remaining missing data

print(X_test[['GarageYrBlt', 'LotFrontage']].describe().transpose())

print("-------------Count of null values in X_test Data------------------")

print(X_test.loc[:, X_test.isnull().any()].isnull().sum())



# Fill null 'GarageYrBlt' values with 1900.

X_train['GarageYrBlt'].fillna(1900, inplace=True)



X_test['GarageYrBlt'].fillna(1900, inplace=True)



# Fill missing 'LotFrontage' with random numbers

X_train.LotFrontage[np.isnan(X_train.LotFrontage)] = np.random.normal(X_train.LotFrontage.mean(), X_train.LotFrontage.std(), np.isnan(X_train.LotFrontage).sum())

X_test.LotFrontage[np.isnan(X_test.LotFrontage)] = np.random.normal(X_test.LotFrontage.mean(), X_test.LotFrontage.std(), np.isnan(X_test.LotFrontage).sum())





# Find stats of remaining missing data

print(X_test[['GarageYrBlt', 'LotFrontage']].describe().transpose())

print("-------------Count of null values in X_test Data------------------")

print(X_test.loc[:, X_test.isnull().any()].isnull().sum())
# Drop rows with outlier data

# [change if required] columns_for_outliers = ['YearBuilt', 'GrLivArea', 'TotalBsmtSF', 'Fireplaces', 'GarageArea', 'LotArea'] 

outliers_iqr(X_train, columns_for_outliers)

print('Total ' + str(X_train.Outlier.sum()) + ' rows with outliers from X_train were deleted')

X_train = X_train[X_train.Outlier != 1]

X_train = X_train.drop(['Outlier'], axis=1)
# Drop rows with outlier data

# Delete more rows from 'LotFrontage' outliers

X_train = X_train[X_train.LotFrontage < 130]
# Create dependent variable dataframe

y_train = np.log(X_train.SalePrice)

X_train = X_train.drop(['SalePrice'], axis=1)

y_train.shape
# Create Id dataframes for train and test data

train_id = train_data.iloc[:, 0]

test_id = test_data.iloc[:, 0]



print(train_id.head())

print(test_id.head())
# Change to category data type

for column_name in X_train.select_dtypes(include=['object']).columns:

    X_train[column_name] = X_train[column_name].astype('category')

    

for column_name in X_test.select_dtypes(include=['object']).columns:

    X_test[column_name] = X_test[column_name].astype('category')

    

# Change categorical columns to category data type

X_train['MSSubClass'] = X_train['MSSubClass'].astype('category')

X_test['MSSubClass'] = X_test['MSSubClass'].astype('category')
# Transform categorical features in to dummy variables

# Get the list of category columns

cat_col_names = X_train.select_dtypes(include=['category']).columns



X_train = pd.get_dummies(X_train, columns=cat_col_names, prefix=cat_col_names, drop_first=True)

X_test = pd.get_dummies(X_test, columns=cat_col_names, prefix=cat_col_names, drop_first=True)
# Ensure dummy columns are created properly. By default, all categorical columns will change to numeric one. So no column with categorical data type will be left.  

print(X_train.select_dtypes(include=['category']).columns)

print(X_test.select_dtypes(include=['category']).columns)
# Check number of columns and name of columns match between X_train and X_test

print(X_train.shape)

print(X_test.shape)

print(set(X_train.columns) == set(X_test.columns))

print('--------columns present in X_train but not in X_test-------')

missing_col_tt = [i for i in list(X_train) if i not in list(X_test)]

print(missing_col_tt)

print('--------columns present in X_test but not in X_train-------')

missing_col_tr = [i for i in list(X_test) if i not in list(X_train)]

print(missing_col_tr)



# Drop these columns and test again

X_train = X_train.drop(missing_col_tt, axis=1)

X_test = X_test.drop(missing_col_tr, axis=1)



print(X_train.shape)

print(X_test.shape)

print(set(X_train.columns) == set(X_test.columns))

print('--------columns present in X_train but not in X_test-------')

missing_col_tt = [i for i in list(X_train) if i not in list(X_test)]

print(missing_col_tt)

print('--------columns present in X_test but not in X_train-------')

missing_col_tr = [i for i in list(X_test) if i not in list(X_train)]

print(missing_col_tr)
# Apply Feature Scaling

fs = MaxAbsScaler()

X_train_fs = fs.fit_transform(X_train)

X_test_fs = fs.transform(X_test)

print(X_train_fs.shape)

print(X_test_fs.shape)

print(X_train_fs[:2,:6])

print(X_test_fs[:2,:6])
lr = LinearRegression()

model = lr.fit(X_train_fs, y_train)

predictions = model.predict(X_train_fs)

actual_values = y_train

R_square = model.score(X_train_fs, y_train)

MSE = mean_squared_error(y_train, predictions)

RMSE = np.sqrt(MSE)

plt.scatter(predictions, actual_values, alpha=.5, color='g') #alpha helps to show overlapping data

plt.xlabel('Predicted Price')

plt.ylabel('Actual Price')

plt.title('Linear Regression Model')

plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2, color='r')

overlay = 'R^2 is: {:.3f}\nMSE is: {:.4f}\nRMSE is: {:.4f}'.format(R_square, MSE, RMSE)

plt.annotate(s=overlay,xy=(12.1,10.6),size='x-large')

plt.show()
# Find cross-validation score

lr2 = LinearRegression()

scores = cross_val_score(estimator = lr2, X = X_train, y = y_train, scoring="neg_mean_squared_error", cv=10)

rmse_scores = np.sqrt(-scores)

def display_scores(scores):

    print("Best RMSE:", scores.min())

    print("Mean RMSE:", scores.mean())

    print("SD RMSE:", scores.std())



display_scores(rmse_scores)



# Generate cross-validated estimates for each input data point

lr1 = LinearRegression()



# cross_val_predict returns an array of the same size as `y` where each entry is a prediction obtained by cross validation:

predictions = cross_val_predict(lr1, X_train, y_train, cv=10)



MSE = mean_squared_error(y_train, predictions)

RMSE = np.sqrt(MSE)



fig, ax = plt.subplots()

ax.scatter(y_train, predictions, alpha=.5, color='g', edgecolors=(0, 0, 0)) #alpha helps to show overlapping data

ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2, color='r')

overlay = 'MSE is: {:.4f}\nRMSE is: {:.4f}'.format(MSE, RMSE)

plt.annotate(s=overlay,xy=(12.1,10.6),size='x-large')



ax.set_xlabel('Predicted Price')

ax.set_ylabel('Actual Price')

plt.title('Cross-validated estimates - Linear Regression')

plt.show()
# Create submission file and make first submission to Kaggle.

lrs = LinearRegression()

model = lrs.fit(X_train, y_train)

y_pred = lrs.predict(X_test)

y_pred = np.exp(y_pred)

print(y_pred)

# Combine Id and prediction

House_price_pred = np.vstack((test_id, y_pred))

# Create output file

np.savetxt('House_price_pred.csv', np.transpose(House_price_pred), delimiter=',', fmt="%s")
# Compare various linear regression algorithms

rs = 1

ests = [ LinearRegression(), Ridge(), Lasso(), ElasticNet(), BayesianRidge(), OrthogonalMatchingPursuit()]



ests_labels = np.array(['Linear', 'Ridge', 'Lasso', 'ElasticNet', 'BayesRidge', 'OMP'])

errvals = np.array([])



X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)



for e in ests:

    e.fit(X_tr, y_tr)

    this_err = mean_squared_error(y_val, e.predict(X_val))

    RMSE = np.sqrt(this_err)

    #print "got error %0.2f" % this_err

    errvals = np.append(errvals, RMSE)



pos = np.arange(errvals.shape[0])

srt = np.argsort(errvals)

plt.figure(figsize=(7,5))

plt.bar(pos, errvals[srt], align='center')

plt.xticks(pos, ests_labels[srt])

plt.xlabel('Estimator')

plt.ylabel('RMSE')

plt.show()
# Generate cross-validated estimates for each input data point

from sklearn.model_selection import cross_val_predict

BRidge = BayesianRidge()



# cross_val_predict returns an array of the same size as `y` where each entry is a prediction obtained by cross validation:

predictions = cross_val_predict(BRidge, X_train, y_train, cv=10)



MSE = mean_squared_error(y_train, predictions)

RMSE = np.sqrt(MSE)



fig, ax = plt.subplots()

ax.scatter(y_train, predictions, alpha=.5, color='g', edgecolors=(0, 0, 0)) #alpha helps to show overlapping data

ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2, color='r')

overlay = 'MSE is: {:.4f}\nRMSE is: {:.4f}'.format(MSE, RMSE)

plt.annotate(s=overlay,xy=(12.1,10.6),size='x-large')



ax.set_xlabel('Predicted Price')

ax.set_ylabel('Actual Price')

plt.title('Cross-validated estimates - Bayesian Ridge')





# Find cross-validation score

scores = cross_val_score(estimator = BRidge, X = X_train, y = y_train, scoring="neg_mean_squared_error", cv=10)

rmse_scores = np.sqrt(-scores)

def display_scores(scores):

    print("Best RMSE:", scores.min())

    print("Mean RMSE:", scores.mean())

    print("SD RMSE:", scores.std())



display_scores(rmse_scores)

plt.show()
# Train and Validation set split by model_selection

X_tr, X_val, y_tr, y_val = train_test_split(X_train_fs, y_train, test_size=0.25, random_state=42)

plt.figure(figsize=(28, 7))

for i in range (-2, 3):

    alpha = 10**i

    rm = Ridge(alpha=alpha)

    ridge_model = rm.fit(X_tr, y_tr)

    preds_ridge = ridge_model.predict(X_val)

    R_square = ridge_model.score(X_val, y_val)

    MSE = mean_squared_error(y_val, preds_ridge)

    RMSE = np.sqrt(MSE)

    

    plt.subplot(1, 5, i+3)

    plt.scatter(preds_ridge, y_val, alpha=.5, color='g', edgecolors=(0, 0, 0))

    plt.xlabel('Predicted Price')

    plt.ylabel('Actual Price')

    plt.title('Ridge Regularization with alpha = {}'.format(alpha))

    overlay = 'R^2 is: {:.3f}\nRMSE is: {:.4f}'.format(R_square, RMSE)

    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2, color='r')

    plt.annotate(s=overlay,xy=(12.1,10.6),size='large')

plt.show()
pr = PolynomialFeatures(degree = 2)

X_poly = pr.fit_transform(X_train_fs)



# Train and Validation set split by model_selection

X_tr, X_val, y_tr, y_val = train_test_split(X_poly, y_train, test_size=0.25, random_state=42)



lr = LinearRegression()

model = lr.fit(X_tr, y_tr)

predictions = model.predict(X_val)

actual_values = y_val



R_square = model.score(X_val, y_val)

MSE = mean_squared_error(y_val, predictions)

RMSE = np.sqrt(MSE)



plt.scatter(predictions, actual_values, alpha=.5, color='g') #alpha helps to show overlapping data

plt.xlabel('Predicted Price')

plt.ylabel('Actual Price')

plt.title('Polynomial Regression Model')

overlay = 'R^2 is: {:.3f}\nRMSE is: {:.4f}'.format(R_square, RMSE)

plt.annotate(s=overlay,xy=(12.1,10.6),size='x-large')

plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2, color='r')

plt.show()
reg_names = ["Linear Regression", "Ridge Regularization", "Nearest Neighbors", "Gaussian Process",

         "Decision Tree", "Neural Net", "Random Forest", "XGBoost", "Gradient Boost", "Extra Tree", "AdaBoost",

         "Bayesian Ridge", "Bagging"]



reg_func = [LinearRegression(normalize=True),

    Ridge(alpha=0.01),

    KNeighborsRegressor(n_neighbors=5),

    GaussianProcessRegressor(alpha =0.01, n_restarts_optimizer=9, normalize_y=True, ),

    DecisionTreeRegressor(max_depth=8),

    MLPRegressor(),

    RandomForestRegressor(n_estimators=410, max_depth=8),

    XGBRegressor(max_depth = 8, n_estimators = 16),

    GradientBoostingRegressor(n_estimators = 500, max_depth = 8, min_samples_split = 2, learning_rate = 0.01, loss ='ls'),

    ExtraTreesRegressor(n_estimators=200, max_depth=8, criterion='mae'),

    AdaBoostRegressor(DecisionTreeRegressor(max_depth=8)),

    BayesianRidge(),

    BaggingRegressor(RandomForestRegressor(n_estimators=410, max_depth=8))]



np.random.seed(42)

                     

# iterate over regressors

print('       Model     --->     Best RMSE  --->   Mean RMSE  --->  SD RMSE')

i=1

for name, regr in zip(reg_names, reg_func):

    mse_scores = cross_val_score(estimator = regr, X = X_train, y = y_train, scoring="neg_mean_squared_error", cv=5)

    rmse_scores = np.sqrt(-mse_scores)

    best_rmse = "{:.3f}".format(rmse_scores.min())

    mean_rmse = "{:.3f}".format(rmse_scores.mean())

    sd_rmse = "{:.3f}".format(rmse_scores.std())

    print(i,' - ', name, ' --->  ',  best_rmse, ' --->  ', mean_rmse, '  --->  ', sd_rmse)

    i += 1
# Train and Validation set split by model_selection

X_tr, X_val, y_tr, y_val = train_test_split(X_train_fs, y_train, test_size=0.25, random_state=42)



# Fit regression model

regr_1 = DecisionTreeRegressor()

regr_2 = DecisionTreeRegressor(max_depth=2)

regr_3 = DecisionTreeRegressor(max_depth=8)

regr_1.fit(X_tr, y_tr)

regr_2.fit(X_tr, y_tr)

regr_3.fit(X_tr, y_tr)



# Predict

y_1 = regr_1.predict(X_val)

y_2 = regr_2.predict(X_val)

y_3 = regr_3.predict(X_val)



S_1 = np.sqrt(mean_squared_error(y_val, y_1))

S_2 = np.sqrt(mean_squared_error(y_val, y_2))

S_3 = np.sqrt(mean_squared_error(y_val, y_3))



# Cross validation scores

scores = cross_val_score(estimator = regr_3, X = X_train, y = y_train, scoring="neg_mean_squared_error", cv=10)

rmse_scores = np.sqrt(-scores)

def display_scores(scores):

    print("Best RMSE:", scores.min())

    print("Mean RMSE:", scores.mean())

    print("SD RMSE:", scores.std())

display_scores(rmse_scores)



# Plot the results

plt.figure(figsize=(12, 8))



plt.scatter(y_val, y_1, s=20, edgecolor="black", c="yellowgreen", label='Default: / RMSE: ' + str("{:.3f}".format(S_1)),  marker = 'o')

plt.scatter(y_val, y_2, s=20, edgecolor="black", c="darkorange", label='max_depth=2 / RMSE: ' + str("{:.3f}".format(S_2)), marker = 'x')

plt.scatter(y_val, y_3, s=20, edgecolor="black", c="cornflowerblue", label='max_depth=8 / RMSE: ' + str("{:.3f}".format(S_3)),  marker = 's')

plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2, color='r')

plt.xlabel("Actual")

plt.ylabel("Prediction")

plt.title("Decision Tree Regression")

plt.legend()

plt.show()
# Train and Validation set split by model_selection

X_tr, X_val, y_tr, y_val = train_test_split(X_train_fs, y_train, test_size=0.25, random_state=42)



# Fit regression model

regr_1 = RandomForestRegressor()

regr_2 = RandomForestRegressor(n_estimators=410)

regr_3 = RandomForestRegressor(max_depth=8)

regr_1.fit(X_tr, y_tr)

regr_2.fit(X_tr, y_tr)

regr_3.fit(X_tr, y_tr)



# Predict

y_1 = regr_1.predict(X_val)

y_2 = regr_2.predict(X_val)

y_3 = regr_3.predict(X_val)



S_1 = np.sqrt(mean_squared_error(y_val, y_1))

S_2 = np.sqrt(mean_squared_error(y_val, y_2))

S_3 = np.sqrt(mean_squared_error(y_val, y_3))



# Cross validation scores

scores = cross_val_score(estimator = regr_2, X = X_train, y = y_train, scoring="neg_mean_squared_error", cv=10)

rmse_scores = np.sqrt(-scores)

def display_scores(scores):

    print("Best RMSE:", scores.min())

    print("Mean RMSE:", scores.mean())

    print("SD RMSE:", scores.std())

display_scores(rmse_scores)



# Plot the results

plt.figure(figsize=(28, 8))

plt.subplot(1, 2, 1)

plt.scatter(y_val, y_1, s=20, edgecolor="black", c="yellowgreen", label='Default / RMSE: ' + str("{:.4f}".format(S_1)),  marker = 'o')

plt.scatter(y_val, y_2, s=20, edgecolor="black", c="darkorange", label='n_estimators=410 / RMSE: ' + str("{:.4f}".format(S_2)), marker = 'x')

plt.scatter(y_val, y_3, s=20, edgecolor="black", c="cornflowerblue", label='max_depth=8 / RMSE: ' + str("{:.4f}".format(S_3)),  marker = 's')

plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2, color='r')

plt.xlabel("Actual")

plt.ylabel("Prediction")

plt.title("Random Forest Regression")

plt.legend()



# Plot feature importance

feature_importance = regr_2.feature_importances_

# make importances relative to max importance

feature_importance = 100.0 * (feature_importance / feature_importance.max())

sorted_idx = np.argsort(feature_importance)

sorted_idx = np.array([i for i in list(sorted_idx) if i < 21])

pos = np.arange(sorted_idx.shape[0]) + .5

plt.subplot(1, 2, 2)

plt.barh(pos, feature_importance[sorted_idx], align='center')

plt.yticks(pos, X_train.columns[sorted_idx])

plt.xlabel('Relative Importance')

plt.title('Variable Importance')

plt.show()
# Train and Validation set split by model_selection

X_tr, X_val, y_tr, y_val = train_test_split(X_train_fs, y_train, test_size=0.25, random_state=42)



# Fit regression model

regr_1 = ExtraTreesRegressor()

regr_2 = ExtraTreesRegressor(n_estimators=200, criterion='mae')

regr_3 = ExtraTreesRegressor(max_depth=8)

regr_1.fit(X_tr, y_tr)

regr_2.fit(X_tr, y_tr)

regr_3.fit(X_tr, y_tr)



# Predict

y_1 = regr_1.predict(X_val)

y_2 = regr_2.predict(X_val)

y_3 = regr_3.predict(X_val)



S_1 = np.sqrt(mean_squared_error(y_val, y_1))

S_2 = np.sqrt(mean_squared_error(y_val, y_2))

S_3 = np.sqrt(mean_squared_error(y_val, y_3))



# Cross validation scores

scores = cross_val_score(estimator = regr_2, X = X_train, y = y_train, scoring="neg_mean_squared_error", cv=10)

rmse_scores = np.sqrt(-scores)

def display_scores(scores):

    print("Best RMSE:", scores.min())

    print("Mean RMSE:", scores.mean())

    print("SD RMSE:", scores.std())

display_scores(rmse_scores)



# Plot the results

plt.figure(figsize=(28, 8))

plt.subplot(1, 2, 1)

plt.scatter(y_val, y_1, s=20, edgecolor="black", c="yellowgreen", label='Default / RMSE: ' + str("{:.4f}".format(S_1)),  marker = 'o')

plt.scatter(y_val, y_2, s=20, edgecolor="black", c="darkorange", label='n_estimators=200 / RMSE: ' + str("{:.4f}".format(S_2)), marker = 'x')

plt.scatter(y_val, y_3, s=20, edgecolor="black", c="cornflowerblue", label='max_depth=8 / RMSE: ' + str("{:.4f}".format(S_3)),  marker = 's')

plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2, color='r')

plt.xlabel("Actual")

plt.ylabel("Prediction")

plt.title("Extra Tree Regression")

plt.legend()



# Plot feature importance

feature_importance = regr_2.feature_importances_

# make importances relative to max importance

feature_importance = 100.0 * (feature_importance / feature_importance.max())

sorted_idx = np.argsort(feature_importance)

sorted_idx = np.array([i for i in list(sorted_idx) if i < 21])

pos = np.arange(sorted_idx.shape[0]) + .5

plt.subplot(1, 2, 2)

plt.barh(pos, feature_importance[sorted_idx], align='center')

plt.yticks(pos, X_train.columns[sorted_idx])

plt.xlabel('Feature Importance')

plt.title('Variable Importance')

plt.show()
# Gradient Boosting regression - cross_val_score

# Train and Validation set split by model_selection

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Fit regression model

params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,

          'learning_rate': 0.01, 'loss': 'ls'}

regr = GradientBoostingRegressor(**params)



regr.fit(X_tr, y_tr)

R_score = regr.score(X_val,y_val)

print("R-squared: %.3f" % R_score)

rmse = np.sqrt(mean_squared_error(y_val, regr.predict(X_val)))

print("RMSE: %.4f" % rmse)



# Cross validation scores

scores = cross_val_score(estimator = regr, X = X_train, y = y_train, scoring="neg_mean_squared_error", cv=10)

rmse_scores = np.sqrt(-scores)

def display_scores(scores):

    print("Best RMSE:", scores.min())

    print("Mean RMSE:", scores.mean())

    print("SD RMSE:", scores.std())

display_scores(rmse_scores)



# Plot training deviance

# compute test set deviance

test_score = np.zeros((params['n_estimators'],), dtype=np.float64)



for i, y_pred in enumerate(regr.staged_predict(X_val)):

    test_score[i] = regr.loss_(y_val, y_pred)



plt.figure(figsize=(28, 8))

plt.subplot(1, 2, 1)

plt.title('Deviance')

plt.plot(np.arange(params['n_estimators']) + 1, regr.train_score_, 'b-',

         label='Training Set Deviance')

plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',

         label='Test Set Deviance')

plt.legend(loc='upper right')

plt.xlabel('Boosting Iterations')

plt.ylabel('Deviance')



# Plot feature importance

feature_importance = regr.feature_importances_

# make importances relative to max importance

feature_importance = 100.0 * (feature_importance / feature_importance.max())

sorted_idx = np.argsort(feature_importance)

sorted_idx = np.array([i for i in list(sorted_idx) if i < 21])

pos = np.arange(sorted_idx.shape[0]) + .5

plt.subplot(1, 2, 2)

plt.barh(pos, feature_importance[sorted_idx], align='center')

plt.yticks(pos, X_train.columns[sorted_idx])

plt.xlabel('Relative Importance')

plt.title('Variable Importance')

plt.show()
# Compare ensemble methods MSEs over range of estimators - Bagging, Random Forest and Adaboost

plt_start = time.time()

# Set the number of estimators and the "step factor" used to plot the graph of MSE for each method

n_jobs = -1  # Parallelisation factor for bagging, random forests

n_estimators = 500

step_factor = 20

axis_step = int(n_estimators/step_factor)



# Train and Validation set split by model_selection

X_tr, X_val, y_tr, y_val = train_test_split(X_train_fs, y_train, test_size=0.3)

    

# Pre-create the arrays which will contain the RMSE for each particular ensemble method

estimators = np.zeros(axis_step)

bagging_rmse = np.zeros(axis_step)

rf_rmse = np.zeros(axis_step)

boosting_rmse = np.zeros(axis_step)



# Estimate the Bagging RMSE over the full number of estimators, across a step size ("step_factor")

for i in range(0, axis_step):

    bagging = BaggingRegressor(DecisionTreeRegressor(), n_estimators=step_factor*(i+1), n_jobs=n_jobs, random_state=42)

    bagging.fit(X_tr, y_tr)

    rmse = np.sqrt(mean_squared_error(y_val, bagging.predict(X_val)))

    estimators[i] = step_factor*(i+1)

    bagging_rmse[i] = rmse



# Estimate the Random Forest MSE over the full number

# of estimators, across a step size ("step_factor")

for i in range(0, axis_step):

    rf = RandomForestRegressor(n_estimators=step_factor*(i+1), n_jobs=n_jobs, random_state=42)

    rf.fit(X_tr, y_tr)

    rmse = np.sqrt(mean_squared_error(y_val, rf.predict(X_val)))

    estimators[i] = step_factor*(i+1)

    rf_rmse[i] = rmse



# Estimate the AdaBoost MSE over the full number

# of estimators, across a step size ("step_factor")

for i in range(0, axis_step):

    boosting = AdaBoostRegressor(DecisionTreeRegressor(), n_estimators=step_factor*(i+1), learning_rate=0.01, random_state=42)

    boosting.fit(X_tr, y_tr)

    rmse = np.sqrt(mean_squared_error(y_val, boosting.predict(X_val)))

    estimators[i] = step_factor*(i+1)

    boosting_rmse[i] = rmse



# Plot the chart of MSE versus number of estimators

plt.figure(figsize=(28, 8))

plt.title('Bagging, Random Forest and Boosting comparison')

plt.plot(estimators, bagging_rmse, 'b-', color="yellowgreen", label='Bagging')

plt.plot(estimators, rf_rmse, 'b-', color="cornflowerblue", label='Random Forest')

plt.plot(estimators, boosting_rmse, 'b-', color="darkorange", label='AdaBoost')

plt.legend(loc='upper right')

plt.xlabel('Estimators')

plt.ylabel('Root Mean Squared Error')

plt.show()

plt_end = time.time()

print('Time taken to plot ML models : ' + str("{:.2f}".format((plt_end - plt_start)/60)) + ' minutes')
one_to_left = ss.beta(10, 1)  

from_zero_positive = ss.expon(0, 50)



params = {  

    "n_estimators": range(3, 40),

    "max_depth": range(3, 40),

    "learning_rate": ss.uniform(0.05, 0.4),

    "colsample_bylevel": one_to_left,

    "subsample": one_to_left,

    "gamma": ss.uniform(0, 10),

    'reg_alpha': from_zero_positive,

    "min_child_weight": from_zero_positive,

}



xgbreg = XGBRegressor(nthreads=-1) 



from sklearn.model_selection import RandomizedSearchCV



gs = RandomizedSearchCV(xgbreg, params, scoring='r2', n_jobs=1)  

gs.fit(X_train_fs, y_train)  

print(gs.best_score_)

print(gs.best_params_)
X_tr_mlr = X_train[['YearBuilt', 'OverallQual', 'GrLivArea', 'TotalBsmtSF', 'Fireplaces', 'GarageYrBlt', 'GarageArea', 'TotalBath', 'LotArea', 'WoodDeckSF',\

                    'OpenPorch', 'LotFrontage', 'MSSubClass_80', 'MSSubClass_40', 'MSSubClass_70']]

X_tr_mlr = MaxAbsScaler().fit_transform(X_tr_mlr) # result remains same even after Feature Scaling

X_tr_mlr = sm.add_constant(X_tr_mlr) ## let's add an intercept (beta_0) to our model

regressor_ols = sm.OLS(endog = y_train, exog = X_tr_mlr).fit()

regressor_ols.summary()
X_tr_mlr = X_train[['YearBuilt', 'OverallQual', 'GrLivArea', 'TotalBsmtSF', 'Fireplaces', 'GarageYrBlt', 'GarageArea', 'TotalBath', 'LotArea', 'WoodDeckSF',\

                    'OpenPorch', 'LotFrontage', 'MSSubClass_80', 'MSSubClass_40', 'MSSubClass_70']]

X_tr_mlr = MaxAbsScaler().fit_transform(X_tr_mlr) # result remains same even after Feature Scaling

lr = LinearRegression()

model = lr.fit(X_tr_mlr, y_train)

predictions = model.predict(X_tr_mlr)



R_square = model.score(X_tr_mlr, y_train)

RMSE = np.sqrt(mean_squared_error(y_train, predictions))



plt.scatter(predictions, y_train, alpha=.5, color='g') #alpha helps to show overlapping data

plt.xlabel('Predicted Price')

plt.ylabel('Actual Price')

plt.title('Linear Regression Model - Selected Features')

plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2, color='r')

overlay = 'R^2 is: {:.3f}\nRMSE is: {:.4f}'.format(R_square, RMSE)

plt.annotate(s=overlay,xy=(12.1,10.6),size='x-large')

plt.show()
# Principal componenet plot without Feature Scaling

from sklearn.decomposition import PCA

pca = PCA(n_components=100)

pca.fit(X_train)

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('Number of components')

plt.ylabel('Cumulative explained variance')
# Principal componenet plot with Feature Scaling

from sklearn.decomposition import PCA

pca = PCA(n_components=60)

pca.fit(X_train_fs)

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('Number of components')

plt.ylabel('Cumulative explained variance')
# See how much variance can be explained by top 10 principal components

cum_exp_var = np.cumsum(pca.explained_variance_ratio_)

cum_exp_var[:10]
# Create output file to check which variables play important role in an specific principal component

pca = PCA(n_components=4)

pca.fit(X_train_fs)

pca_dist = pd.DataFrame(pca.components_,columns=X_train.columns,index = ['PC-1','PC-2','PC-3', 'PC-4'])

filename = 'pca_dist.csv'

pca_dist.to_csv(filename, index=False, encoding='utf-8')

pca_dist
# Gradient Boosting Out-of-Bag estimates

'''# Check wheter any of the element is NaN or infinite and then clean

print(np.any(np.isnan(X_train_fs)))

print(np.any(np.isfinite(X_train_fs)))



def clean_dataset(df):

    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"

    df.dropna(inplace=True)

    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)

    return df[indices_to_keep].astype(np.float64)

print(np.any(np.isnan(X_train_fs)))

print(np.any(np.isfinite(X_train_fs)))

clean_dataset(pd.DataFrame(X_train_fs))'''



# Train and Validation set split by model_selection

X_tr, X_val, y_tr, y_val = train_test_split(np.array(X_train), np.array(y_train), test_size=0.5, random_state = 42)



# Fit Regressor with out-of-bag estimates

params = {'n_estimators': 1200, 'max_depth': 3, 'subsample': 0.5,

          'learning_rate': 0.01, 'min_samples_leaf': 1, 'random_state': 3}

regr = GradientBoostingRegressor(**params)



regr.fit(X_tr, y_tr)

acc = regr.score(X_val, y_val)

print("Accuracy: {:.4f}".format(acc))



n_estimators = params['n_estimators']

x = np.arange(n_estimators) + 1





def heldout_score(regr, X_val, y_val):

    """compute deviance scores on ``X_val`` and ``y_val``. """

    score = np.zeros((n_estimators,), dtype=np.float64)

    for i, y_pred in enumerate(regr.staged_predict(X_val)):

        score[i] = regr.loss_(y_val, y_pred)

    return score





def cv_estimate(n_splits=3):

    cv = KFold(n_splits=n_splits)

    cv_regr = GradientBoostingRegressor(**params)

    val_scores = np.zeros((n_estimators,), dtype=np.float64)

    for train, test in cv.split(X_tr, y_tr):

        cv_regr.fit(X_tr[train], y_tr[train])

        val_scores += heldout_score(cv_regr, X_tr[test], y_tr[test])

    val_scores /= n_splits

    return val_scores





# Estimate best n_estimator using cross-validation

cv_score = cv_estimate(3)



# Compute best n_estimator for test data

test_score = heldout_score(regr, X_val, y_val)



# negative cumulative sum of oob improvements

cumsum = -np.cumsum(regr.oob_improvement_)



# min loss according to OOB

oob_best_iter = x[np.argmin(cumsum)]



# min loss according to test (normalize such that first loss is 0)

test_score -= test_score[0]

test_best_iter = x[np.argmin(test_score)]



# min loss according to cv (normalize such that first loss is 0)

cv_score -= cv_score[0]

cv_best_iter = x[np.argmin(cv_score)]



# color brew for the three curves

oob_color = list(map(lambda x: x / 256.0, (190, 174, 212)))

test_color = list(map(lambda x: x / 256.0, (127, 201, 127)))

cv_color = list(map(lambda x: x / 256.0, (253, 192, 134)))



# plot curves and vertical lines for best iterations

plt.plot(x, cumsum, label='OOB loss', color=oob_color)

plt.plot(x, test_score, label='Test loss', color=test_color)

plt.plot(x, cv_score, label='CV loss', color=cv_color)

plt.axvline(x=oob_best_iter, color=oob_color)

plt.axvline(x=test_best_iter, color=test_color)

plt.axvline(x=cv_best_iter, color=cv_color)



# add three vertical lines to xticks

xticks = plt.xticks()

xticks_pos = np.array(xticks[0].tolist() +

                      [oob_best_iter, cv_best_iter, test_best_iter])

xticks_label = np.array(list(map(lambda t: int(t), xticks[0])) +

                        ['OOB', 'CV', 'Test'])

ind = np.argsort(xticks_pos)

xticks_pos = xticks_pos[ind]

xticks_label = xticks_label[ind]

plt.xticks(xticks_pos, xticks_label)



plt.legend(loc='upper right')

plt.ylabel('normalized loss')

plt.xlabel('number of iterations')



plt.show()
param_grid = {

    "n_estimators": [ 100, 300, 500, 600 ],

    "max_depth" : [ 3, 5, 8, 10 ],

    "learning_rate": [ .001, .01, .1 ],

    "max_features" : [ 'auto', 'sqrt', 10 ],

    "loss" : [ 'ls', 'lad' ]

}

          

gbr = GradientBoostingRegressor(random_state=42)

# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 

grid_search = GridSearchCV(gbr, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs = -1)

grid_search.fit(X_train, y_train)



# The best hyperparameter combination found:

print('Best Parameters: ', grid_search.best_params_)

print('Best RMSE score: ', np.sqrt(-grid_search.best_score_))

print('Best Estimators: ', grid_search.best_estimator_)
# Find best estimator

best = grid_search.best_estimator_
# Create submission file and make first submission to Kaggle

model = best.fit(X_train, y_train)

y_pred = best.predict(X_test)

y_pred = np.exp(y_pred)

print(y_pred)

# Combine Id and prediction

House_price_pred = np.vstack((test_id, y_pred))

# Create output file

np.savetxt('House_price_pred_final.csv', np.transpose(House_price_pred), delimiter=',', fmt="%s")
end = time.time()

print('Total running time of the script : ', str("{:.2f}".format((end - start)/60)), ' minutes')