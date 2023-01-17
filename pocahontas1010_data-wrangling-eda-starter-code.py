import numpy as np # linear algebra

import pandas as pd 

import os

print(os.listdir("../input"))

import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



# Graphics in SVG format are more sharp and legible

%config InlineBackend.figure_format = 'svg' 



# Increase the default plot size and set the color scheme

plt.rcParams['figure.figsize'] = 8, 5

plt.rcParams['image.cmap'] = 'viridis'



%matplotlib inline
pd.set_option('display.max_rows', 5000)

pd.set_option('display.max_columns', 5000)
# load the dataset

PATH_TO_DATA = '../input'



df_train = pd.read_csv(os.path.join(PATH_TO_DATA, 

                                             'train.csv'), index_col='Id')

df_test = pd.read_csv(os.path.join(PATH_TO_DATA, 

                                             'test.csv'), index_col='Id')
# take a look at the first 5 rows of the dataset

df_train.head()
# take a look at the first 5 rows of the dataset

df_test.head()
# number of samples and columns

df_train.shape, df_test.shape
# check for duplicates

sum(df_train.duplicated()), sum(df_test.duplicated())
# check the datatypes

df_train.info()
# column names

df_train.columns
# continuous variables

continuous_features = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 

                       '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',

                       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']



df_train[continuous_features].head()
# the number of continuous features

df_test[continuous_features].shape[1]
# discrete variables

discrete_features = ['YearBuilt', 'YearRemodAdd', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 

                     'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'MoSold', 'YrSold']

# check the filter

df_train[discrete_features].head()
# the number of discrete features

df_test[discrete_features].shape[1]
# nominal variables

nominal_features = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'Utilities', 'LotConfig', 'Neighborhood', 'Condition1', 

                    'Condition2', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',

                    'Foundation', 'Heating', 'CentralAir', 'Electrical', 'GarageType', 'PavedDrive', 'MiscFeature',

                    'SaleType', 'SaleCondition']



# check the filter

df_train[nominal_features].head()
# the number of continuous features

df_test[nominal_features].shape[1]
# ordinal variables

ordinal_features = ['LotShape', 'LandContour', 'LandSlope', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 

                    'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

                    'HeatingQC', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 

                    'PoolQC', 'Fence']



# check the filter

df_train[ordinal_features].head()
# the number of continuous features

df_test[ordinal_features].shape[1]
# test to check if all the columns are included

list(continuous_features + discrete_features + nominal_features + ordinal_features).sort() == list(df_test.columns).sort() 
# check for missing values

df_train.isnull().sum()[df_train.isnull().sum() > 0]
# the number 

nuls_columns = list(df_train.isnull().sum()[df_train.isnull().sum() > 0].index)

len(nuls_columns)
# nuls in continuous features

continuous_nuls = [nul_columns for nul_columns in nuls_columns if nul_columns in continuous_features]

continuous_nuls
# nuls in discrete features

discrete_nuls = [nul_columns for nul_columns in nuls_columns if nul_columns in discrete_features]

discrete_nuls
# nuls in nominal features

nominal_nuls = [nul_columns for nul_columns in nuls_columns if nul_columns in nominal_features]

nominal_nuls
# nuls in ordinal features

ordinal_nuls = [nul_columns for nul_columns in nuls_columns if nul_columns in ordinal_features]

ordinal_nuls
# check for missing values

df_test.isnull().sum()[df_test.isnull().sum() > 0]
# the number of variables with nulls

nuls_columns = list(df_test.isnull().sum()[df_test.isnull().sum() > 0].index)

len(nuls_columns)
# nuls in continuous features

continuous_nuls = [nul_columns for nul_columns in nuls_columns if nul_columns in continuous_features]

continuous_nuls
# nuls in discrete features

discrete_nuls = [nul_columns for nul_columns in nuls_columns if nul_columns in discrete_features]

discrete_nuls
# nuls in nominal features

nominal_nuls = [nul_columns for nul_columns in nuls_columns if nul_columns in nominal_features]

nominal_nuls
# nuls in ordinal features

ordinal_nuls = [nul_columns for nul_columns in nuls_columns if nul_columns in ordinal_features]

ordinal_nuls
# let's filter only for discrete, nominal and ordinal features

unique_filter =  discrete_features + nominal_features + ordinal_features
# non-null unique values for ordinal features

df_train[ordinal_features].nunique()
# non-null unique values for nominal features

df_train[nominal_features].nunique()
# non-null unique values for nominal features

df_train[discrete_features].nunique()
df_train[discrete_features].head()
# non-null unique values differences between training and testing set

df_diff_features = df_train[unique_filter].nunique() - df_test[unique_filter].nunique()

df_diff_features = df_diff_features[df_diff_features != 0]

df_diff_features
df_diff_features.plot(kind='barh', figsize=(10, 10));

plt.title('Categorical Features Differences Training/Testing Set')

plt.show()
# describe the dataset

df_train[continuous_features + ['SalePrice']].describe().T
for feat in ['Alley', 'GarageType', 'MiscFeature', 'FireplaceQu', 'Fence']:

    # fill NaNs

    df_train[feat].fillna(f'No{feat}', inplace=True)

    df_test[feat].fillna(f'No{feat}', inplace=True)

    print(f'{feat}...done')
# test for training set

df_train[['Alley', 'GarageType', 'MiscFeature', 'FireplaceQu', 'Fence']].isnull().sum()
# check for testing set 

df_test[['Alley', 'GarageType', 'MiscFeature', 'FireplaceQu', 'Fence']].isnull().sum()
# fill for no basement

for feat in ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']:

    # fill NaNs

    df_train[feat].fillna(f'NoBasement', inplace=True)

    df_test[feat].fillna(f'NoBasement', inplace=True)

    print(f'{feat}...done')
# test for train

df_train[['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']].isnull().sum()
# and testing set

df_test[['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']].isnull().sum()
# fill for no pool

df_train['PoolQC'].fillna(f'NoPool', inplace=True)

df_test['PoolQC'].fillna(f'NoPool', inplace=True)
df_train['PoolQC'].isnull().sum(), df_test['PoolQC'].isnull().sum()
# fill for no garage

for feat in ['GarageFinish', 'GarageQual', 'GarageCond']:

    # fill NaNs

    df_train[feat].fillna(f'NoGarage', inplace=True)

    df_test[feat].fillna(f'NoGarage', inplace=True)

    print(f'{feat}...done')
# test for train

df_train[['GarageFinish', 'GarageQual', 'GarageCond']].isnull().sum()
# and testing set

df_test[['GarageFinish', 'GarageQual', 'GarageCond']].isnull().sum()
# check again for missing values

df_test[nominal_features+ordinal_features].isnull().sum()[df_test.isnull().sum() > 0]
# check for missing values

df_train[nominal_features+ordinal_features].isnull().sum()[df_train.isnull().sum() > 0]
# let's see the entries

df_test[df_test['MSZoning'].isnull()]
df_test.loc[df_test[df_test['MSZoning'].isnull()].index, 'MSZoning']
# index variable

MSZoning_null_index = list(df_test[df_test['MSZoning'].isnull()].index)
df_test.loc[MSZoning_null_index[0], 'MSZoning'] = 'I'

df_test.loc[MSZoning_null_index[1], 'MSZoning'] = 'A'

df_test.loc[MSZoning_null_index[2], 'MSZoning'] = 'A'

df_test.loc[MSZoning_null_index[3], 'MSZoning'] = 'I'
# test the changes

df_test['MSZoning'].isnull().sum()
df_train['MSZoning'].value_counts()
df_test['MSZoning'].value_counts()
# reassign values

df_test.loc[MSZoning_null_index[0], 'MSZoning'] = 'RH'

df_test.loc[MSZoning_null_index[1], 'MSZoning'] = 'RL'

df_test.loc[MSZoning_null_index[2], 'MSZoning'] = 'RL'

df_test.loc[MSZoning_null_index[3], 'MSZoning'] = 'RH'
# plot categorical feature differences between training and testing set

def plot_bar(feature):

    width = 0.35

    ind = np.arange(df_test[feature].value_counts().shape[0])

    locations = ind + width / 2 # ytick locations

    labels = list(df_test[feature].value_counts().index) # ytick labels



    heights_test = list(df_test[feature].value_counts().values)

    heights_train = list(df_train[feature].value_counts().values)

    plot_test = plt.bar(ind, heights_test, width, label='Test')

    plot_train = plt.bar(ind + width, heights_train, width, label='Train')



    plt.title('{} Bar Chart'.format(feature))

    plt.xlabel('{}'.format(feature))

    plt.ylabel('')

    plt.xticks(locations, labels)



    plt.legend()

    plt.show()
plot_bar('MSZoning')
df_test[df_test['Utilities'].isnull()]
# index variable

Utilities_null_index = list(df_test[df_test['Utilities'].isnull()].index)

Utilities_null_index
# assign the new values

df_test.loc[Utilities_null_index, 'Utilities'] = 'NoSewr'
# test the changes

df_test['Utilities'].isnull().sum()
plot_bar('Utilities')
df_test['Utilities'].value_counts()
df_train['Utilities'].value_counts()
df_test[df_test['Exterior1st'].isnull()]
# index variable

Exterior1st_null_index = df_test[df_test['Exterior1st'].isnull()].index[0]
# reassign values

df_test.loc[Exterior1st_null_index, 'Exterior1st'] ='PreCast'

df_test.loc[Exterior1st_null_index, 'Exterior2nd'] ='PreCast'

df_test.loc[Exterior1st_null_index, 'GarageYrBlt'] ='NoGarage'
df_test[df_test['Exterior2nd'].isnull()].shape
df_test[df_test['SaleType'].isnull()]
# index variable

SaleType_null_index = df_test[df_test['SaleType'].isnull()].index[0]
df_test.loc[SaleType_null_index, 'SaleType'] ='VWD'
df_test[df_test['SaleType'].isnull()]
df_test['SaleType'].value_counts()
df_train['SaleType'].value_counts()
# I'll put it in the Oth category in order to have similar structure

df_test.loc[SaleType_null_index, 'SaleType'] ='Oth'
plot_bar('SaleType')
df_test[df_test['KitchenQual'].isnull()]
# index variable

KitchenQual_null_index = df_test[df_test['KitchenQual'].isnull()].index[0]
#df_test.loc[KitchenQual_null_index, 'KitchenQual'] = 'Po'

# reassign value to match distributions

df_test.loc[KitchenQual_null_index, 'KitchenQual'] = 'Fa'
df_test['KitchenQual'].value_counts()
df_train['KitchenQual'].value_counts()
df_test[df_test['Functional'].isnull()]
# index variable

Functional_null_index = list(df_test[df_test['Functional'].isnull()].index)
df_test.loc[Functional_null_index[0], 'Functional'] = 'Sev'

df_test.loc[Functional_null_index[1], 'Functional'] = 'Sev'
df_train['Functional'].value_counts()
df_test['Functional'].value_counts()
plot_bar('Functional')
df_train[df_train['Electrical'].isnull()]
# index variable

Electrical_null_index = df_train[df_train['Electrical'].isnull()].index[0]
df_train.loc[Electrical_null_index, 'Electrical'] = 'SBrkr'
df_train['Electrical'].value_counts()
df_test['Electrical'].value_counts()
df_train[df_train['MasVnrType'].isnull()]
df_test[df_test['MasVnrType'].isnull()]
# index variable

MasVnrType_null_index = list(df_train[df_train['MasVnrType'].isnull()].index)

MasVnrType_null_index
# assign values

df_train.loc[MasVnrType_null_index, 'MasVnrArea'] = 0

df_train.loc[MasVnrType_null_index, 'MasVnrType'] = 'None'
MasVnrType_null_index = list(df_test[df_test['MasVnrType'].isnull()].index)



df_test.loc[MasVnrType_null_index, 'MasVnrArea'] = 0

df_test.loc[MasVnrType_null_index, 'MasVnrType'] = 'None'
plot_bar('MasVnrType')
df_test[continuous_features + discrete_features].isnull().sum()[df_test[continuous_features + discrete_features].isnull().sum() > 0]
df_train[continuous_features + discrete_features].isnull().sum()[df_train[continuous_features + discrete_features].isnull().sum() > 0]
# Fill LotFrontage with mean

df_train['LotFrontage'].describe()
plt.hist(df_train['LotFrontage'], bins=30, alpha=0.5, label='Train set')

plt.hist(df_test['LotFrontage'], bins=30, alpha=0.5, label='Test set')



plt.title("LotFrontage Histogram Train/Test")

plt.xlabel('LotFrontage ($ft$)')

plt.ylabel('Frequency')



plt.legend()

plt.show()
# median value LotFrontage

LotFrontage_null_fill = df_train['LotFrontage'].mode()[0]

# fill nans for training and testing set

df_train['LotFrontage'].fillna(LotFrontage_null_fill, inplace=True)

df_test['LotFrontage'].fillna(LotFrontage_null_fill, inplace=True)
df_train['LotFrontage'].isnull().sum(), df_test['LotFrontage'].isnull().sum()
df_train[df_train['GarageYrBlt'].isnull()][:10]
GarageYrBlt_null_vals_train = list(df_train[df_train['GarageYrBlt'].isnull()].index)

GarageYrBlt_null_vals_test = list(df_test[df_test['GarageYrBlt'].isnull()].index)
df_train.loc[GarageYrBlt_null_vals_train, 'GarageYrBlt'] = 'NoGarage'

df_test.loc[GarageYrBlt_null_vals_test, 'GarageYrBlt'] = 'NoGarage'
df_train['GarageYrBlt'].isnull().sum(), df_test['GarageYrBlt'].isnull().sum()
df_test[continuous_features + discrete_features].isnull().sum()[df_test[continuous_features + discrete_features].isnull().sum() > 0]
df_test[df_test['GarageArea'].isnull()]
GarageArea_null_index = df_test[df_test['GarageArea'].isnull()].index[0]
df_test.loc[GarageArea_null_index, 'GarageCars'] = 0

df_test.loc[GarageArea_null_index, 'GarageArea'] = 0
df_test[df_test['BsmtFullBath'].isnull()]
BsmtFullBath_nulls_index = list(df_test[df_test['BsmtFullBath'].isnull()].index)

mask_bsm = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',  'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']

df_test.loc[BsmtFullBath_nulls_index, mask_bsm] = 0
df_test[mask_bsm].isnull().sum()
# check to see if we cleaned for nulls

df_train.isnull().sum()[df_train.isnull().sum() > 0], df_test.isnull().sum()[df_train.isnull().sum() > 0]
# let's see the four observations with GrLivArea bigger the 4000 

# from the training set

df_test[df_test['GrLivArea'] > 4000]
# let's see the four observations with GrLivArea bigger the 4000 

# from the training set

df_train[df_train['GrLivArea'] > 4000]
# drop the rows with extreme values

df_train.drop(df_train[df_train['GrLivArea'] > 4000].index, inplace=True)
# test the change

df_train[df_train['GrLivArea'] > 4000]
# reset index

df_train = df_train.reset_index(drop=True)

# check the shape of our dataframe

df_train.shape
# rename the index column

df_train.index.names = ['Id'] 
df_train.head()
# save this for later

df_train.to_csv(os.path.join('train_clean.csv'), sep=',', index_label='Id')

df_test.to_csv(os.path.join('test_clean.csv'), sep=',', index_label='Id')
df_train.head()
df_train = pd.read_csv(os.path.join( 

                                             'train_clean.csv'), index_col='Id')

df_test = pd.read_csv(os.path.join( 

                                             'test_clean.csv'), index_col='Id')

df_train.head()
plt.hist(df_train['SalePrice'], bins=30)

plt.title("Sale Price Histogram")

plt.xlabel('SalePrice ($USD$)')

plt.ylabel('Frequency')



plt.show()
plt.hist(np.log(df_train['SalePrice']), bins=30)

plt.title("Sale Price (Log Transformation) Histogram")

plt.xlabel('$log(SalePrice)$ ($USD$)')

plt.ylabel('Frequency')



plt.show()
plt.hist(df_train['GrLivArea'], bins=30)

plt.title("Above Ground Living Area Square Feet Histogram")

plt.xlabel('GrLivArea ($ft^{2}$)')

plt.ylabel('Frequency')



plt.show()
plt.scatter(df_train.GrLivArea, df_train.SalePrice)

plt.title("Above Ground Living Area Square Feet vs Sale Price (training set)")

plt.xlabel('GrLivArea')

plt.ylabel('Sale Price')

plt.show()
plt.hist(df_train['GrLivArea'], bins=30, alpha=0.5, label='Train set')

plt.hist(df_test['GrLivArea'], bins=30, alpha=0.5, label='Test set')



plt.title("Above Ground Living Area Square Feet Histogram Train/Test")

plt.xlabel('GrLivArea ($ft^{2}$)')

plt.ylabel('Frequency')



plt.legend()

plt.show()
# group data by year and month

df_time_price = df_train.groupby(['YrSold', 'MoSold'], as_index=False)['SalePrice'].mean()

# see the first rows

df_time_price.head()
mean_2010 = df_time_price[df_time_price['YrSold'] == 2010]['SalePrice'].mean()

mean_2010
# create data with median value

new_dummy_df2010 = pd.DataFrame(np.array([[2010, 8, mean_2010], [2010, 9, mean_2010], [2010, 10, mean_2010], 

                       [2010, 11, mean_2010], [2010, 12, mean_2010]]),

                   columns=['YrSold', 'MoSold', 'SalePrice'], index=[55, 56, 57, 58, 59])

new_dummy_df2010
# append the new data

df_time_price = df_time_price.append(new_dummy_df2010)
years = list(range(2006,2011))

labels = list(range(1, 13))
plt.figure(figsize=(8, 6))



for year in years:

    plt.plot(labels, df_time_price[df_time_price['YrSold'] == year]['SalePrice'], label=year)



plt.title('Sale Price During Time Period')

plt.xticks(labels)

plt.xlabel('Months')

plt.ylabel('Sale Price')

plt.legend()

plt.show()
# calculate correlation matrix

corr = df_train[continuous_features + ['SalePrice']].corr()

#Plot figsize

fig, ax = plt.subplots(figsize=(12, 12))

#Generate Color Map

colormap = sns.diverging_palette(220, 10, as_cmap=True)

#Generate Heat Map, allow annotations and place floats in map

g = sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")

#Apply xticks

# plt.xticks(range(len(corr.columns)), corr.columns);

# #Apply yticks

# plt.yticks(range(len(corr.columns)), corr.columns)

# #show plot

plt.show()
imp_cont_features = ['GrLivArea', 'TotalBsmtSF', 'GarageArea', 'MasVnrArea', 'LotFrontage', 

                     'OpenPorchSF', 'WoodDeckSF']
%config InlineBackend.figure_format = 'png' 

sm = pd.plotting.scatter_matrix(df_train[imp_cont_features + ['SalePrice']], figsize=(30, 30), diagonal='kde');



for ax in sm.ravel():

    ax.set_xlabel(ax.get_xlabel(), fontsize = 20, rotation = 45)

    ax.set_ylabel(ax.get_ylabel(), fontsize = 20, rotation = 0)



#May need to offset label when rotating to prevent overlap of figure

[s.get_yaxis().set_label_coords(-0.5,0.5) for s in sm.reshape(-1)]



#Hide all ticks

[s.set_xticks(()) for s in sm.reshape(-1)]

[s.set_yticks(()) for s in sm.reshape(-1)]

plt.show()
for feat in imp_cont_features:

    plt.scatter(df_train[feat], np.log(df_train.SalePrice))

    plt.xlabel(feat)

    plt.ylabel('$log(Sale Price)$')

    plt.show()
df_train['TotalSquareFootage'] = df_train['GrLivArea'] + df_train['TotalBsmtSF']

df_test['TotalSquareFootage'] = df_test['GrLivArea'] + df_test['TotalBsmtSF']
plt.scatter(df_train['TotalSquareFootage'], np.log(df_train.SalePrice))

plt.xlabel('TotalSquareFootage')

plt.ylabel('$log(Sale Price)$')

plt.show()
df_train['logSalePrice'] = np.log(df_train.SalePrice)
# calculate correlation matrix

corr = df_train[['TotalSquareFootage', 'SalePrice']].corr()

#Plot figsize

fig, ax = plt.subplots(figsize=(3,3))

#Generate Color Map

colormap = sns.diverging_palette(220, 10, as_cmap=True)

#Generate Heat Map, allow annotations and place floats in map

g = sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")

#Apply xticks

# plt.xticks(range(len(corr.columns)), corr.columns);

# #Apply yticks

# plt.yticks(range(len(corr.columns)), corr.columns)

# #show plot

plt.show()
import statsmodels.api as sm
df_train['intercept'] = 1
X = df_train[['intercept', 'TotalSquareFootage']]

y = df_train['SalePrice']
# predicting the price and add all of our var that are quantitative

lm = sm.OLS(y, X)

results = lm.fit()

results.summary()
# these are our cofficients for our function

np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(), X)) , X.transpose()), y)
df_train['Neighborhood'].value_counts()
# create neighborhood dummies

neighborhood_dummies = pd.get_dummies(df_train['Neighborhood'])

neighborhood_dummies.head()
# select all the columns but the first

neighborhood_columns = list(neighborhood_dummies.columns[1:])

neighborhood_dummies[neighborhood_columns].head()
X = X.join(neighborhood_dummies)

X.head()
y = df_train['SalePrice']
lm2 = sm.OLS(y, X[['intercept', 'TotalSquareFootage'] + neighborhood_columns])

results2 = lm2.fit()

results2.summary()
'{0:.10f}'.format(-4.119e+04)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error
X.drop(columns=['intercept'], inplace=True)

y = df_train['logSalePrice']
reg = LinearRegression()

# fit training data

reg.fit(X, y)

# get the R^2

reg.score(X, y)
# get the coefficients

reg.coef_
# get the intercept

reg.intercept_
# make predictions

pred = reg.predict(X)
# calculate RMSE

def rmse(y, pred):

    return np.sqrt(mean_squared_error(y, pred))
# error

rmse(y, pred)
# calculate RMSE

# np.sqrt(mean_squared_error(y, pred))
# load the dataset

PATH_TO_DATA = '../input'



sub = pd.read_csv(os.path.join(PATH_TO_DATA, 

                                             'sample_submission.csv'), index_col='Id')
df_test.head()
# create neighborhood dummies

neighborhood_dummies_test = pd.get_dummies(df_test['Neighborhood'])

neighborhood_dummies_test.head()
X_test = df_test[['TotalSquareFootage']]

X_test = X_test.join(neighborhood_dummies_test)

X_test.head()
# make predictions

pred_test = reg.predict(X_test)

# exponentiate the results

pred_test = np.exp(pred_test)

pred_test[:10]
sub['SalePrice'] = pred_test
plt.hist(pred_test, bins=40);

plt.title('Distribution of SalePrice predictions');
sub.to_csv('model1.csv')

# load the dataset



model1_sub = pd.read_csv(os.path.join( 

                                             'model1.csv'), index_col='Id')

model1_sub.head() # 0.19363