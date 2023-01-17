import numpy as np
import pandas as pd
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))

# Statistical packages used for transformations
from scipy import stats
from scipy.stats import skew, norm
from scipy.special import boxcox1p    # 博克斯- 考克斯变换，又称幂变换

import matplotlib.pyplot as plt
import matplotlib.gridspec as mg
import seaborn as sns
sns.set(style='white', context='notebook', palette='deep')

import sklearn.preprocessing as sp

# Algorithms used for modeling
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import Lasso
import xgboost as xgb

# Model selection packages for sampling dataset and optimising parameters
from sklearn import model_selection
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

import sklearn.linear_model as lm
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import voting_classifier
from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
# Load data
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
# Merge the all input data
dataset = pd.concat(objs=[train_data.iloc[:,:-1], test_data.iloc[:,:]], axis=0).reset_index(drop=True)

print('train_data.shape = ', train_data.shape)
print('test_data.shape = ', test_data.shape)
dataset.shape
train_data.columns
train_data.head()
print(train_data.info())
# There are five observations that an instructor may wish to remove from the data set before 
# giving it to students (a plot of SALE PRICE versus GR LIV AREA will quickly indicate these points)

# salesprice vs GrLivArea --> Above grade (ground) living area square feet
plt.subplots(figsize=(15,5))

plt.subplot(1,2,1)
g = sns.scatterplot(x='GrLivArea', y='SalePrice', hue=None, data=train_data, color='deepskyblue')
g.set_title('Before Filter')

# drop the outliers: filter and delete data with GrLivArea>4000
outliers_index = train_data[train_data['GrLivArea']>4000].index
print(outliers_index)

# update the train_data
train_data = train_data.drop(outliers_index, axis=0)
print(train_data.shape)

plt.subplot(1,2,2)
g = sns.scatterplot(x='GrLivArea', y='SalePrice', hue=None, data=train_data, color='deepskyblue')
g = g.set_title('After Filter')
# Updata all the related data
dataset = dataset.drop(outliers_index, axis=0)

# Save the length of the train and test data
len_train = len(train_data)
len_test = len(test_data)
print('(len_train len_test) --> (',len_train, len_test,')')

# Save the ID column
train_ID = train_data['Id']
test_ID = test_data['Id']

# Save the output y for train_data
y_train = train_data['SalePrice']
print('y_train.shape --> ', y_train.shape)

# Drop the ID for data as it is redundant for modeling
train_data.drop('Id', axis=1, inplace=True)
test_data.drop('Id', axis=1, inplace=True)
dataset.drop('Id', axis=1, inplace=True)
print('train_data.shape --> ',train_data.shape)
print('test_data.shape --> ',test_data.shape)
print('dataset.shape -->', dataset.shape)
# For dataset: Aggreate the missing values
dataset.isnull().sum()

# Features including missing values or nan values for dataset
data_IncludeNa = dataset.isnull().sum()[dataset.isnull().sum() != 0].sort_values(ascending=False)
data_NaIn_index = data_IncludeNa.index
print('data_NaIn_index number:',len(data_NaIn_index))
print(data_NaIn_index)
data_NaIn_numeric = dataset[data_NaIn_index]._get_numeric_data()
print(data_NaIn_numeric.columns)
plt.subplots(figsize=(15, 10))
g = sns.barplot(x=data_NaIn_index, y=data_IncludeNa)
# 设置刻度，一定是针对坐标轴plt.gca()进行设置，其参数是barplot的坐标轴标签
g = plt.gca().set_xticklabels(g.get_xticklabels(), rotation=90)
# 或者 -->  fig = data_IncludeNa.sort_values(ascending=False).plot(kind='bar')
# According to the data_description file, fill missing values with 'None'
data_Na_None_fts =  ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu","GarageType", 
                     "GarageFinish", "GarageQual", "GarageCond","BsmtQual", "BsmtCond", 
                     "BsmtExposure", "BsmtFinType1","BsmtFinType2", "MSSubClass", "MasVnrType"]
dataset[data_Na_None_fts] = dataset[data_Na_None_fts].fillna('None')
print("Fill missing values with 'None' --> Done")
# The next step is to fill missing values for other features.
data_Na_unk_fts = list(set(data_NaIn_index) - set(data_Na_None_fts))
print(data_Na_unk_fts, '-->', len(data_Na_unk_fts))
# For features in data_Na_unk_fts, according to the data description and the value_counts of 
# each features, features below could be filled wiht '0' directly:
data_Na_zero_fts = ["GarageYrBlt", "GarageArea", "GarageCars", "BsmtFinSF1","BsmtFinSF2", 
                    "BsmtUnfSF", "TotalBsmtSF", "MasVnrArea","BsmtFullBath", "BsmtHalfBath"]
dataset[data_Na_zero_fts] = dataset[data_Na_zero_fts].fillna(0)
print("Fill missing values with 'zero' --> Done")
# The remaining features with missing values
data_Na_unk1_fts = list(set(data_Na_unk_fts) - set(data_Na_zero_fts))
print(data_Na_unk1_fts, '-->', len(data_Na_unk1_fts))
# The missing values number for the remaining features after fillna with None and Zero
data_IncludeNa[data_Na_unk1_fts]
# For those with just few missing values, it can be filled with the mode
# Series.mode()[0] == Series.value_counts()[0]
data_Na_mode_fts = ['MSZoning', 'Electrical', 'Exterior2nd', 'Exterior1st', 'Functional', 'KitchenQual', 'SaleType']
for feature in data_Na_mode_fts:
    # dataset[feature] = dataset[feature].fillna(dataset[dataset[feature].notnull()][feature].mode()[0])
    dataset[feature] = dataset[feature].fillna(dataset[feature].value_counts()[0])
dataset[data_Na_mode_fts].isnull().sum()
print("Fill missing values with 'mode' --> Done")
# For last feature LotFrontage, it means the 'linear feet of street connected to property'.
# (the same street area(feature:Neighborhood) basicly have the same linear feet to street)
# It could be estimated via the neighborhood's data
print(dataset[['Neighborhood', 'LotFrontage']].sort_values('Neighborhood').head(13))
# median value for grouped data :  sever as dict  df.to_dict()
group_dict = dataset[['Neighborhood','LotFrontage']].groupby('Neighborhood').median().to_dict()
fill_value_dict = list(group_dict.values())[0]
print(fill_value_dict)
# For LotFrontage, it's filled with the median value groupby Neighborhood

# Method 1: 1) fill_func;   2)groupby -> ['feature'] --> .apply(fill_func)
fill_func = lambda g:g.fillna(g.median())
dataset['LotFrontage'] = dataset.groupby('Neighborhood', group_keys=False)['LotFrontage'].apply(fill_func)
print("Fill missing values with 'Group Mode' --> Done")

# Method 2:
# fill_func = lambda g:g.fillna(fill_value_dict[g.name])         # g.name 分组的name属性
# dataset['LotFrontage'] = dataset.groupby('Neighborhood', group_keys=False)['LotFrontage'].apply(fill_func)

# Method 3: .transform(func)
# All are same with Method 1 except replacing apply to transform
# feature condition with missing values
dataset.isnull().sum()[dataset.isnull().sum() !=0]
train_data.Utilities.isnull().sum()
test_data.Utilities.isnull().sum()
print(train_data.Utilities.value_counts(),'\n\n', test_data.Utilities.value_counts())
# Drop Utilities feature
dataset = dataset.drop('Utilities', axis=1)
dataset.shape
# Below is operated considering the Categorization in numerical features in feature engineering 

# GarageYrBlt is float, convert its type to int for binning
dataset['GarageYrBlt'] = dataset['GarageYrBlt'].astype('int')

# GarageCar is float, convert its type to int for dummies
print(dataset['GarageCars'].unique())
dataset['GarageCars'] = dataset['GarageCars'].astype('int')
# feature condition with missing values
dataset.isnull().sum()[dataset.isnull().sum() !=0]
# Updata train and test data based on dataset
# For train-data, since it has been cutted from dataset without SalePrice, it has to concat it.
train_data = pd.concat([dataset.iloc[:len_train, :], y_train], axis=1)
test_data = dataset.iloc[len_train:, :]

print(train_data.shape, len(test_data), len(train_ID), len(test_ID))
train_data.isnull().sum().value_counts()
# Obtain the numerical features and categrical features for train_data

# numerical columns method
numerical_data = train_data._get_numeric_data()
print('Numerical Features Number:', numerical_data.shape[1])
print('Categrical Features Number:', train_data.shape[1] - numerical_data.shape[1])
numerical_data.head()
# corr among numerical features
corr_numerical = numerical_data.corr()
corr_numerical.shape
# heatmap
plt.subplots(figsize=(14, 10))
g = sns.heatmap(corr_numerical, cmap='coolwarm', annot=False)
# Extract the top features 
# nlargest" Get the rows of a DataFrame sorted by the `n` largest values of `columns`.
#k = 11
#features_top10 = corr_numerical.nlargest(k, 'SalePrice').index

# corr > 0.5
features_top = corr_numerical[corr_numerical['SalePrice']>0.5].index
# Obtain the correlation among top features
corr_top = corr_numerical.loc[features_top, features_top]
corr_top.SalePrice
# Check the type of top corr features
dataset[corr_top.index[:-1]].dtypes
plt.subplots(figsize=(12, 7))
g = sns.heatmap(corr_top, cmap='coolwarm', annot=True, fmt='.2f')
g = sns.pairplot(train_data[features_top], diag_kind='kde')
# Update Numerical data for dataset
numerical_data = dataset._get_numeric_data()
numerical_features = list(numerical_data.columns)
print(len(numerical_features),'\n',numerical_features)
# Non-numerical features in dataset
non_numerical_features = list(set(dataset.columns) - set(numerical_data.columns))
print(len(non_numerical_features), '\n', non_numerical_features)
dataset.select_dtypes('object').columns
# Since SalePrice is our target, exploring relationship with SalePrice needs to operate in train_data

# Correlation among all the numerical features in train_data
numeric_corr_train = train_data.corr()
# top : corr > 0.5
#numeric_feature_train_top = train_data[train_data.SalePrice[numeric_corr_train > 0.5]]
numeric_corr_train_top = numeric_corr_train[numeric_corr_train['SalePrice'] > 0.5].SalePrice.sort_values(ascending=False)
numeric_corr_train_top
# Features top 
numeric_feature_top = numeric_corr_train_top.index
print(numeric_feature_top.values)
# Explore the corr matrix
plt.subplots(figsize=(14,7))
g = sns.heatmap(numeric_corr_train.loc[numeric_feature_top, numeric_feature_top], cmap='coolwarm', annot=True)
g = plt.gca().set_xticklabels(g.get_xticklabels(), rotation=60)
# Explore OverallQual vs SalePrice
# g = sns.factorplot(x='OverallQual', y='SalePrice', data=train_data, kind='bar')
plt.subplots(figsize=(23,6))
plt.subplot(1,3,1)
g = sns.barplot(train_data.OverallQual,train_data.SalePrice,palette='coolwarm')
plt.subplot(1,3,2)
g = sns.stripplot(train_data.OverallQual,train_data.SalePrice,palette='coolwarm')
plt.subplot(1,3,3)
g = sns.boxplot(train_data.OverallQual,train_data.SalePrice,palette='coolwarm')
plt.subplots(figsize=(23,6))
plt.subplot(1,2,1)
g = sns.lineplot(x='YearRemodAdd', y='SalePrice', data=train_data)
plt.subplot(1,2,2)
g = sns.distplot(dataset.YearRemodAdd, bins=25)
# Define the time difference, explore its relationship with saleprice
train_data['Remod_Diff'] = train_data['YearRemodAdd'] - train_data['YearBuilt']

plt.subplots(figsize=(40,20))
plt.subplot(2,1,2)
g = sns.lineplot(x='Remod_Diff', y='SalePrice', data=train_data, color='limegreen')
plt.subplot(2,1,1)
g = sns.lineplot(x='YearRemodAdd', y='SalePrice', data=train_data, color='purple')
g = sns.lineplot(x='YearBuilt', y='SalePrice', data=train_data,color='orange')
# Creat a new feature to replace feature YearRemodAdd
dataset['Remod_Diff'] = dataset['YearRemodAdd'] - dataset['YearBuilt']
dataset.drop('YearRemodAdd', axis=1, inplace=True)
plt.subplots(figsize=(23,6))
plt.subplot(1,2,1)
g = sns.lineplot(x='YearBuilt', y='SalePrice', data=train_data)
plt.subplot(1,2,2)
g = sns.distplot(dataset.YearBuilt, bins=25)
# Bining for all dataset

# Firstly, creat a new feature
dataset['YearBuilt_Band'] = pd.cut(dataset.YearBuilt, 8)
dataset['YearBuilt_Band'].unique()
# 可以采用该方法，但columns名为 YearBuilt_(1871.862, 1889.25]，不够简洁
#dummies = pd.get_dummies(dataset['YearBuilt_Band'], prefix='YearBuilt')
#dummies.head()
#dataset = dataset.join(dummies)
#dataset.head()
# Second, name the bins
dataset.loc[dataset.YearBuilt <= 1890, 'YearBuilt'] = 1
dataset.loc[(dataset.YearBuilt > 1890) & (dataset.YearBuilt <= 1906), 'YearBuilt'] = 2
dataset.loc[(dataset.YearBuilt > 1906) & (dataset.YearBuilt <= 1923), 'YearBuilt'] = 3
dataset.loc[(dataset.YearBuilt > 1923) & (dataset.YearBuilt <= 1941), 'YearBuilt'] = 4
dataset.loc[(dataset.YearBuilt > 1941) & (dataset.YearBuilt <= 1958), 'YearBuilt'] = 5
dataset.loc[(dataset.YearBuilt > 1958) & (dataset.YearBuilt <= 1975), 'YearBuilt'] = 6
dataset.loc[(dataset.YearBuilt > 1975) & (dataset.YearBuilt <= 1992), 'YearBuilt'] = 7
dataset.loc[dataset.YearBuilt > 1992, 'YearBuilt'] = 8
# Third, convert the type
dataset['YearBuilt'] = dataset['YearBuilt'].astype(int)
dataset.YearBuilt.dtype
# Forth, drop the band fenture and convert to dummy matrix
dataset.drop('YearBuilt_Band', axis=1, inplace=True)
dataset = pd.get_dummies(dataset, columns=['YearBuilt'], prefix='YearBuilt')
# Check the converted features' new column name
dataset.columns[dataset.columns.str.contains('YearBuilt')].unique()
# Explore TotalBsmtSF vs SalePrice
plt.subplots(figsize=(16,7))
plt.subplot(1,2,1)
g = sns.scatterplot(train_data.TotalBsmtSF,train_data.SalePrice, color='coral')
plt.subplot(1,2,2)
g = sns.distplot(dataset['TotalBsmtSF'], color='royalblue',
                 label='skew:{:.2f}'.format(skew(dataset['TotalBsmtSF'])))
g = g.legend(loc='best')
g = sns.scatterplot(dataset['1stFlrSF'],dataset['TotalBsmtSF'], color='c')
# Bin TotalBsTotalBsmtSFmtSF -- CREATE A NEW FEATURE
dataset['TotalBsmtSF_Band'] = pd.cut(dataset['TotalBsmtSF'], 10)
dataset['TotalBsmtSF_Band'].unique()
dataset.loc[dataset['TotalBsmtSF'] <= 509, 'TotalBsmtSF'] = 1
dataset.loc[(dataset.TotalBsmtSF > 509) & (dataset.TotalBsmtSF <= 1019), 'TotalBsmtSF'] = 2
dataset.loc[(dataset.TotalBsmtSF > 1019) & (dataset.TotalBsmtSF <= 1528), 'TotalBsmtSF'] = 3
dataset.loc[(dataset.TotalBsmtSF > 1528) & (dataset.TotalBsmtSF <= 2038), 'TotalBsmtSF'] = 4
dataset.loc[(dataset.TotalBsmtSF > 2038) & (dataset.TotalBsmtSF <= 2547), 'TotalBsmtSF'] = 5
dataset.loc[(dataset.TotalBsmtSF > 2547) & (dataset.TotalBsmtSF <= 3057), 'TotalBsmtSF'] = 6
dataset.loc[(dataset.TotalBsmtSF > 3057) & (dataset.TotalBsmtSF <= 3566), 'TotalBsmtSF'] = 7
dataset.loc[dataset.TotalBsmtSF > 3566, 'TotalBsmtSF'] = 8
# Third, convert the type
dataset['TotalBsmtSF'] = dataset['TotalBsmtSF'].astype(int)
dataset.TotalBsmtSF.value_counts()
# Forth, drop the band fenture and convert to dummy matrix
dataset.drop('TotalBsmtSF_Band', axis=1, inplace=True)
dataset = pd.get_dummies(dataset, columns=['TotalBsmtSF'], prefix='TotalBsmtSF')
# Check the converted columns
dataset.columns[dataset.columns.str.contains('TotalBsmtSF')].unique()
# Explore TotalBsmtSF vs SalePrice
plt.subplots(figsize=(16,7))
plt.subplot(1,2,1)
g = sns.scatterplot(train_data['1stFlrSF'],train_data['SalePrice'], color='lightcoral')
plt.subplot(1,2,2)
g = sns.distplot(dataset['1stFlrSF'], color='mediumpurple',
                 label='skew:{:.2f}'.format(skew(dataset['1stFlrSF'])))
g = g.legend(loc='best')
# adjust the skew of feature 1stFlrSF
train_data['1stFlrSF'] = np.log1p(train_data['1stFlrSF'])
dataset['1stFlrSF'] = np.log1p(dataset['1stFlrSF'])

plt.subplots(figsize=(16,7))
plt.subplot(1,2,1)
g = sns.scatterplot(train_data['1stFlrSF'],train_data['SalePrice'], color='lightcoral')
g.set_title('skewed')
plt.subplot(1,2,2)
g = sns.distplot(dataset['1stFlrSF'], color='mediumpurple',
                 label='skew:{:.2f}'.format(skew(dataset['1stFlrSF'])))
g.set_title('skewed')
g = g.legend(loc='best')
# Bin TotalBsmtSF and 1stFlrSF
dataset['1stFlrSF_Band'] = pd.cut(dataset['1stFlrSF'], 6)
dataset['1stFlrSF_Band'].unique()
dataset.loc[(dataset['1stFlrSF'] > 5.811) & (dataset['1stFlrSF'] <= 6.268), '1stFlrSF'] = 1
dataset.loc[(dataset['1stFlrSF'] > 6.268) & (dataset['1stFlrSF'] <= 6.721), '1stFlrSF'] = 2
dataset.loc[(dataset['1stFlrSF'] > 6.721) & (dataset['1stFlrSF'] <= 7.175), '1stFlrSF'] = 3
dataset.loc[(dataset['1stFlrSF'] > 7.175) & (dataset['1stFlrSF'] <= 7.629), '1stFlrSF'] = 4
dataset.loc[(dataset['1stFlrSF'] > 7.629) & (dataset['1stFlrSF'] <= 8.083), '1stFlrSF'] = 5
dataset.loc[(dataset['1stFlrSF'] > 8.083) & (dataset['1stFlrSF'] <= 8.536), '1stFlrSF'] = 6
dataset['1stFlrSF'] = dataset['1stFlrSF'].astype(int)
dataset['1stFlrSF'].value_counts()
# Drop the band fenture and convert to dummy matrix
dataset.drop('1stFlrSF_Band', axis=1, inplace=True)
dataset = pd.get_dummies(dataset, columns=['1stFlrSF'], prefix='1stFlrSF')

# Check the converted columns
dataset.columns[dataset.columns.str.contains('1stFlrSF')].unique()
plt.subplots(figsize=(16,7))
plt.subplot(1,2,1)
g = sns.scatterplot(train_data['GrLivArea'], train_data['SalePrice'], color='lightcoral')
plt.subplot(1,2,2)
g = sns.distplot(dataset['GrLivArea'], color='mediumpurple',
                 label='skew:{:.2f}'.format(skew(dataset['GrLivArea'])))
g = g.legend(loc='best')
# feature_band
dataset['GrLivArea_Band'] = pd.cut(dataset['GrLivArea'], 6)
dataset['GrLivArea_Band'].unique()
# Reassign the value
dataset.loc[dataset['GrLivArea']<=1127.5, 'GrLivArea'] = 1
dataset.loc[(dataset['GrLivArea']>1127.5) & (dataset['GrLivArea']<=1921), 'GrLivArea'] = 2
dataset.loc[(dataset['GrLivArea']>1921) & (dataset['GrLivArea']<=2714.5), 'GrLivArea'] = 3
dataset.loc[(dataset['GrLivArea']>2714.5) & (dataset['GrLivArea']<=3508), 'GrLivArea'] = 4
dataset.loc[(dataset['GrLivArea']>3508) & (dataset['GrLivArea']<=4301.5), 'GrLivArea'] = 5
dataset.loc[dataset['GrLivArea']>4301.5, 'GrLivArea'] = 6
dataset['GrLivArea'] = dataset['GrLivArea'].astype(int)

dataset.drop('GrLivArea_Band', axis=1, inplace=True)

dataset = pd.get_dummies(dataset, columns = ["GrLivArea"], prefix="GrLivArea")
dataset.columns[dataset.columns.str.contains('GrLivArea')].unique()
plt.subplots(figsize=(40,12))
plt.subplot(2,4,1)
g = sns.barplot(train_data['FullBath'], train_data['SalePrice'])
plt.subplot(2,4,2)
g = sns.stripplot(train_data['FullBath'], train_data['SalePrice'])
plt.subplot(2,4,3)
g = sns.barplot(train_data['HalfBath'], train_data['SalePrice'])
plt.subplot(2,4,4)
g = sns.stripplot(train_data['HalfBath'], train_data['SalePrice'])
plt.subplot(2,4,5)
g = sns.barplot(train_data['BsmtFullBath'], train_data['SalePrice'])
plt.subplot(2,4,6)
g = sns.stripplot(train_data['BsmtFullBath'], train_data['SalePrice'])
plt.subplot(2,4,7)
g = sns.barplot(train_data['BsmtHalfBath'], train_data['SalePrice'])
plt.subplot(2,4,8)
g = sns.stripplot(train_data['BsmtHalfBath'], train_data['SalePrice'])
dataset['TotalBathrooms'] = dataset['BsmtHalfBath'] + dataset['BsmtFullBath'] + dataset['HalfBath'] + dataset['FullBath']
train_data['TotalBathrooms'] = train_data['BsmtHalfBath'] + train_data['BsmtFullBath'] + train_data['HalfBath'] + train_data['FullBath']

dataset['TotalBathrooms'] = dataset['TotalBathrooms'].astype(int)
train_data['TotalBathrooms'] = train_data['TotalBathrooms'].astype(int)

plt.subplots(figsize=(16,7))
plt.subplot(1,2,1)
g = sns.barplot(train_data['TotalBathrooms'], train_data['SalePrice'])
plt.subplot(1,2,2)
g = sns.stripplot(train_data['TotalBathrooms'], train_data['SalePrice'])
plt.subplots(figsize=(16,7))
plt.subplot(1,2,1)
g = sns.barplot(train_data['TotRmsAbvGrd'], train_data['SalePrice'])
plt.subplot(1,2,2)
g = sns.stripplot(train_data['TotRmsAbvGrd'], train_data['SalePrice'])
plt.subplots(figsize=(16,7))
plt.subplot(1,2,1)
g = sns.barplot(train_data['GarageCars'], train_data['SalePrice'])
plt.subplot(1,2,2)
g = sns.stripplot(train_data['GarageCars'], train_data['SalePrice'])
plt.subplots(figsize=(16,7))
plt.subplot(1,2,1)
g = sns.scatterplot(train_data.GarageArea,train_data.SalePrice, color='coral')
plt.subplot(1,2,2)
g = sns.distplot(dataset['GarageArea'], color='royalblue',
                 label='skew:{:.2f}'.format(skew(dataset['GarageArea'])))
g = g.legend(loc='best')
dataset['GarageArea_Band'] = pd.cut(dataset['GarageArea'], 4)
dataset['GarageArea_Band'].unique()
dataset.loc[dataset['GarageArea'] <= 372.0, 'GarageArea'] = 1
dataset.loc[(dataset['GarageArea'] > 372.0) & (dataset['GarageArea'] <= 744.0), 'GarageArea'] = 2
dataset.loc[(dataset['GarageArea'] > 744.0) & (dataset['GarageArea'] <= 1116.0), 'GarageArea'] = 3
dataset.loc[dataset['GarageArea'] > 1116.0, 'GarageArea'] = 4

dataset['GarageArea'] = dataset['GarageArea'].astype(int)
dataset['GarageArea'].value_counts()
dataset.drop('GarageArea_Band', axis=1, inplace=True)
# Dummy
dataset = pd.get_dummies(dataset, columns = ["GarageArea"], prefix="GarageArea")
dataset.columns[dataset.columns.str.contains('GarageArea')].unique()
print(dataset.MSSubClass.unique())
print(dataset.OverallCond.unique())
print(dataset.MoSold.unique())
print(dataset.YrSold.unique())
#print(dataset.YearBuilt.unique())
#print(dataset.YearRemodAdd.unique())
#print(dataset.GarageYrBlt.unique())
feature_list_numcat = ['MSSubClass']
feature_list_numbin = ['GarageYrBlt']
feature_list_numlab = ['OverallCond','MoSold','YrSold']
# Feature_list_numcat: MSSubClass
# numerical --> object
dataset['MSSubClass'] = dataset['MSSubClass'].astype('category')
dataset.MSSubClass.head()
# Creat OneHotEncode
dataset = pd.get_dummies(dataset, columns=['MSSubClass'], prefix='MSSubClass')
# Checkout the dummied features (Series.str.contains(..))
dataset.columns[dataset.columns.str.contains('MSS')]
# 2) Feature_list_numbin: GarageYrBlt
# Explore the distribution of GarageYrBlt

indice = dataset['GarageYrBlt'][dataset['GarageYrBlt'] !=0].index           # No garage's data
year_max_garage = dataset['GarageYrBlt'][dataset['GarageYrBlt'] !=0].max()  # 2207, unfinished
year_min_garage = dataset['GarageYrBlt'][dataset['GarageYrBlt'] !=0].min()  # 1895
year_span = year_max_garage - year_min_garage                               # 312
indice_train = indice[:len_train]

plt.subplots(figsize=(22,7))
plt.subplot(1,3,1)
g = sns.distplot(dataset['GarageYrBlt'])
plt.subplot(1,3,2)
g = sns.distplot(dataset['GarageYrBlt'][dataset.GarageYrBlt !=0])
plt.subplot(1,3,3)
g = sns.scatterplot(x='GarageYrBlt', y='SalePrice', data=train_data.loc[indice_train, :])
plt.subplots(figsize =(40, 20))
sns.lineplot(x="GarageYrBlt", y="SalePrice", data=train_data.loc[indice_train, :], color='red');
dataset['GarageYrBlt_Band'] = 0
dataset.loc[indice, 'GarageYrBlt_Band'] = pd.cut(dataset['GarageYrBlt'][indice], 6)   # 2756 length
dataset['GarageYrBlt_Band'].unique()
dataset.loc[(dataset['GarageYrBlt'] > 0) & (dataset['GarageYrBlt'] <= 1947.0), 'GarageYrBlt'] = 1
dataset.loc[(dataset['GarageYrBlt'] >1947.0) & (dataset['GarageYrBlt'] <= 1999.0), 'GarageYrBlt'] = 2
dataset.loc[(dataset['GarageYrBlt'] >1999.0) & (dataset['GarageYrBlt'] <= 2051.0), 'GarageYrBlt'] = 3
dataset.loc[dataset['GarageYrBlt'] >2051, 'GarageYrBlt'] = 4

dataset['GarageYrBlt'] = dataset['GarageYrBlt'].astype(int)
dataset['GarageYrBlt'].value_counts()
dataset.drop('GarageYrBlt_Band', axis=1, inplace=True)
# Dummy
dataset = pd.get_dummies(dataset, columns = ["GarageYrBlt"], prefix="GarageYrBlt")
dataset.columns[dataset.columns.str.contains('GarageYrBlt')].unique()
print(numeric_feature_top.values)
# feature_list_numlab = [OverallCond','MoSold','YrSold']

# Label Encode for feature_list_numlab
def label_encode(feature_list,data=None):
    print('LabelEncode for {} start!'.format(feature_list))
    for item in feature_list:
        encoder = sp.LabelEncoder()
        encoder.fit(data[item])
        data[item] = encoder.transform(data[item])
        print('LabelEncode for feature {} has been done!'.format(item))
    print('LabelEncode for {} has finished!'.format(feature_list))
    return data
# feature_list_numlab_1 = 
label_encode(feature_list_numlab, data=dataset)
print(dataset[feature_list_numlab].head())
dataset.MoSold.unique(), dataset.MoSold.dtype
# The original value of MoSold is integer ranging from 1 to 12, which means the month
dataset.dtypes.value_counts()
# The remaining numerical features untreated.
R_num_feats = list(set(numerical_features)-set(numeric_feature_top[1:])
                   -set(feature_list_numcat)-set(feature_list_numlab)-set(feature_list_numbin))
print(len(R_num_feats),R_num_feats)
# Explore the skew of these features
skewness_R_num = dataset[R_num_feats].apply(lambda g:skew(g)).sort_values(ascending=False)
skewness_R_num
plt.figure(figsize=(20,30))
rows = 7
cols = 3
for row in range(rows):
    for col in range(cols):
        k = row * 3 + col + 1
        plt.subplot(rows, cols, k)
        g = sns.distplot(dataset[R_num_feats].iloc[:,k-1], bins=30)
rnumeric_binary_features = ['MiscVal', 'BsmtFinSF2', 'LowQualFinSF', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'EnclosedPorch']
rnumeric_skewed_features = ['WoodDeckSF','2ndFlrSF','BsmtUnfSF','MasVnrArea','LotFrontage','LotArea','OpenPorchSF','BsmtFinSF1']
for item in rnumeric_binary_features:
    dataset[item] = dataset[item].map(lambda x: 0 if x == 0 else 1)
    print('Binarization for numerical feature {} has been done!'.format(item))
print('Binarization for {} have been done!'.format(rnumeric_binary_features))
lam = 0.2
for item in rnumeric_skewed_features:
    skew1 = skew(dataset[item])
    dataset[item] = boxcox1p(dataset[item], lam)
    skew2 = skew(dataset[item])
    print('Adjusting skew for numerical feature {} has been done!'.format(item))
    print('Skew:{:.2f} ---> {:.2f}'.format(skew1, skew2))
print('Adjusting skew for {} have been done!'.format(rnumeric_skewed_features))
# For BsmtUnfSF, skewness needs to be adjusted again
skew1 = skew(dataset['BsmtUnfSF'])
dataset['BsmtUnfSF'] = boxcox1p(dataset['BsmtUnfSF'], 2.5)
skew2 = skew(dataset['BsmtUnfSF'])
print('Skew:{:.2f} ---> {:.2f}'.format(skew1, skew2))
# For LotFrontage, skewness needs to be adjusted again
skew1 = skew(dataset['LotFrontage'])
dataset['LotFrontage'] = boxcox1p(dataset['LotFrontage'], 2.2)
skew2 = skew(dataset['LotFrontage'])
print('Skew:{:.2f} ---> {:.2f}'.format(skew1, skew2))
dataset[R_num_feats].head()
dataset.dtypes.value_counts()
# type of category
dataset[non_numerical_features].dtypes.value_counts()
# Convert to dummy matrix
dataset = pd.get_dummies(dataset, columns=non_numerical_features)
print(dataset.shape)
dataset.head()
dataset.dtypes.value_counts()
# Explore the distrubution of target variable
plt.subplots()
g = sns.distplot(y_train, label='skew: {:.2f}'.format(skew(y_train)))
g = g.legend(loc='best')
# Norm the distribution: log method
y_train_norm = np.log1p(y_train)
plt.subplots()
g = sns.distplot(y_train_norm, 
                 fit=norm,
                 label='skew: {:.2f}'.format(skew(y_train_norm)))
g = g.legend(loc='best')
for i,j  in zip(np.expm1(y_train_norm)[:5], y_train):
    print('{:.2f} --> {:.2f} ({:.2f})'.format(i,j, i-j))
dataset.shape
# input train data:
x_train = dataset.iloc[:len_train, :]
# input test data
x_test = dataset.iloc[len_train:, :]

y_train = y_train_norm
y_train.shape, x_train.shape
print(len_train, len_test, len(dataset), len(test_ID), len(train_ID))
print(len(x_train), len(x_test))
# Algorithms used for modeling
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import  RandomForestRegressor

# Model selection packages for sampling dataset and optimising parameters
from sklearn import model_selection
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import xgboost as xgb
# xgboosting --> feature_importances_
model = xgb.XGBRegressor()
model.fit(x_train, y_train)

# feature_importances_ 
fi_xgb = model.feature_importances_
# 将序排列索引 indices和对应feature name
indices = np.argsort(fi_xgb)[::-1]
feature_fi_xgb = x_train.columns[indices]

# 重要性排序前80的特征索引及对应的feature name
indices_top = indices[:80]

# visualize the features sorted by feature_importances_
plt.subplots(figsize=(20,20))
g = sns.barplot(x=fi_xgb[indices_top], y=feature_fi_xgb[indices_top],orient='h')
# PLS NOTICE HOW IT IS USED.

from sklearn.feature_selection import SelectFromModel
xgb_feat_sel = SelectFromModel(model, prefit=True)

# Reduce X to the selected features
x_train_xgb = xgb_feat_sel.transform(x_train)
x_test_xgb = xgb_feat_sel.transform(x_test)

# Reduce x_test for target prediction dataset
# x_test = xgb_feat_sel.transform(x_test)
# show the features selected
print('features numbers selected:',x_test_xgb.shape)
# Rebuild the dataset by split method
x_train_f, x_test_f, y_train_f, y_test_f = model_selection.train_test_split(x_train_xgb,y_train,test_size=0.3,random_state=42)
print('x_train_f: ', x_train_f.shape, '\nx_test_f: ', x_test_f.shape, '\ny_train_f: ', y_train_f.shape, '\ny_test_f: ', y_test_f.shape)
pd.concat([pd.DataFrame(x_test_f), y_test_f],axis=1).head()
lmr = lm.LinearRegression()
kr_reg = KernelRidge()
lasso_reg = Lasso()
ridge_reg = lm.Ridge()
ENet_reg = Pipeline([('robust', RobustScaler()), 
                     ('ElasticNet', ElasticNet())])
gboost_reg = GradientBoostingRegressor()
model_list = [lmr, kr_reg, lasso_reg, ridge_reg, ENet_reg, gboost_reg]
cols = ['model','param', 'score']
gsCV_before_data = pd.DataFrame(columns=cols)
gsCV_after_data = pd.DataFrame(columns=cols)

# Define func of cross validation
def RMSE_cv(model, x, y, N):
    # KFold(n_splits=., shuffle=., random_state=.)  K折交叉验证迭代器
    kfold = KFold(n_splits=N, shuffle=True, random_state=7)
    cv_score = cross_val_score(model,x,y,cv=kfold,scoring='neg_mean_squared_error')
    RMSE = np.sqrt(-cv_score)
    return RMSE

def gscv_before_func(model_list,df,x,y,N):
    for i,model in enumerate(model_list):
        # obtain model name: model.__class__.__name__
        df.loc[i,'model'] = model.__class__.__name__
        # obtain model params: model.get_params()
        df.loc[i,'param'] = str(model.get_params())
        # obtain model score
        df.loc[i,'score'] = RMSE_cv(model,x,y,N).mean()
    return df
gsCV_before_data = gscv_before_func(model_list, gsCV_before_data, x_train_f,y_train_f,5)
gsCV_before_data.iloc[4,0] = 'ElasticNet'
gsCV_before_data
gsCV_before_data.iloc[5,1]
# lookup the detail params of model:
def lookup(integer):
    modelname = list(gsCV_before_data.model)
    if integer < len(modelname):
        return modelname[integer],gsCV_before_data[gsCV_before_data.model == modelname[integer]].iloc[0,1]
    else:
        print('The integre is out of the length of modelname list')
lookup(3)
eval(gsCV_before_data.iloc[4,1])['steps'][1]
# define the GridSearchCV function
def gsCV(model, params, cv):
    model_opt = GridSearchCV(model, 
                             params, 
                             refit=True, 
                             scoring = 'neg_mean_squared_error', 
                             cv=cv)
    return model_opt
# Create the param list for each model
# [lmr, kr_reg, lasso_reg, ridge_reg, ENet_reg, gboost_reg]
param_gsCV_lmr = {}

param_gsCV_kr = {'alpha':[0.001,0.003,0.01,0.03], 
                 'kernel':['polynomial','rbf'],
                 'coef0':[10],
                 'degree':[1,3],
                 'gamma':[None]}

param_gsCV_lasso = {'alpha':[0.0001,0.0002,0.0004,0.0008,0.001]}

param_gsCV_ridge = {'alpha': [3,3.3,3.6,3.9,4.2], 'solver': ['cholesky','sparse_cg','lsqr','sag']}

param_gsCV_ENet = [{'robust__quantile_range': [(25.0,75.0)]},
                   {'ElasticNet__alpha':[0.1,0.3,1.0,3,9],
                   'ElasticNet__l1_ratio':[0.2,0.4,0.6,0.8,0.9],
                   'ElasticNet__selection': ['cyclic','random']}]
# ``l1_ratio = 0`` the penalty is an L2 penalty
# ``For l1_ratio = 1`` it is an L1 penalty -- equal with Lasso

param_gsCV_gboost = {'learning_rate': [0.01,0.03,0.1,0.3], 
                     'loss':['ls', 'huber'], 
                     'max_depth': [3,5], 
                     'min_samples_leaf': [1,2],
                     'min_samples_split': [2], 
                     'n_estimators': [100]}

param_gsCV_list = [param_gsCV_lmr, param_gsCV_kr, param_gsCV_lasso, 
                   param_gsCV_ridge,param_gsCV_ENet,param_gsCV_gboost]
lookup(5)
aa = gsCV(lmr, param_gsCV_lmr, 5)
aa.fit(x_train_f,y_train_f)
print(aa.best_estimator_.__class__.__name__)
print(aa.best_params_)
aa.best_estimator_
len(model_list)
model_best_list = []
for i, model, param  in zip(range(len(model_list)), model_list, param_gsCV_list):
    print('Process {} is running!'.format(i))
    # GridSearchCV
    model_opt = gsCV(model, param, 5)
    model_opt.fit(x_train_f,y_train_f)
    
    # model best x_train_f:
    model_best = model_opt.best_estimator_
    model_best_name = model_best.__class__.__name__
    model_best_param = model_opt.best_params_
    model_best_score = np.sqrt(-model_opt.best_score_)
    model_best_list.append(model_best)
    # save to dataframe
    gsCV_after_data.loc[i,'model'] = model_best_name
    gsCV_after_data.loc[i,'param'] = str(model_best_param)
    gsCV_after_data.loc[i,'score'] = model_best_score
    # add best_model to dataframe
    gsCV_after_data.loc[i,'best_model'] = model_best
    
    # best model predict for x_test_f (actual y: y_test_f)
    y_test_f_pred = model_best.predict(x_test_f)
    print('Process {} has completed the prediction task.'.format(i))

    # add RSME for the predict result to dataframe 
    rsme = np.sqrt(np.power(y_test_f - y_test_f_pred, 2).mean())
    gsCV_after_data.loc[i,'rsme'] = round(rsme, 2)
    print('rsme of moedl {} is {}: '.format(model_best_name,rsme))
    if i == 5:
        print(gsCV_after_data.loc[i,'best_model'])
# gsCV_before_data = gscv_before_func(model_list,gsCV_before_data, x_train_f, y_train_f, 5)
gsCV_after_data = gscv_before_func(model_best_list, gsCV_after_data, x_train_f, y_train_f, 5)
gsCV_before_data
model_best_list[-1]
gsCV_after_data.iloc[4,0] = 'ElasticNet'
gsCV_after_data

# select 5 models from 6 models as the stack base_estimator
indice = list(gsCV_after_data.sort_values(by=['score'], ascending=False).index)
model_stack_arr = np.array(model_best_list)[indice]
model_stack = model_stack_arr[:-1]
model_stack
cols_stack = list(gsCV_after_data.iloc[indice,:].model[:-1])
cols_stack
stack_train = pd.DataFrame(columns=cols_stack)
stack_test = pd.DataFrame(columns=cols_stack)
# x_test_f predict to form a new train set: 
# x: x_test_f_1,x_test_f_2,x_test_f_3; y: y_test_f
for i, model in enumerate(model_stack):
    # new train_x dataset stacked
    stack_train.iloc[:,i] = model.predict(x_test_f)
    
    # new test dataset stacked: target data_x to predict final y 
    stack_test.iloc[:,i] = model.predict(x_test_xgb)
stack_train.index, y_test_f.index, stack_train.shape,y_test_f.shape
# if they form a dataframe, the index should be reset.
y_test_f = y_test_f.reset_index(drop=True)
stack_train = pd.concat([stack_train, y_test_f],axis=1)
stack_train.head()
stack_test.head()
# set the new train x and y for new stack data
x_new_train = stack_train.iloc[:,:-1]
y_new_train = stack_train.iloc[:,-1:]
# set new test x in order to predict
x_new_test = stack_test

x_new_train.shape, y_new_train.shape,x_new_test.shape
# model
model_top1 = model_stack_arr[-1]
print(model_top1)
model_final = make_pipeline(RobustScaler(), model_top1)
print(model_final)
# model train
model_final.fit(x_new_train, y_new_train)
# model predict
y_final = pd.DataFrame(np.expm1(model_final.predict(x_new_test)))
print(y_final.head())
y_df = pd.DataFrame()
for i, model in enumerate(model_stack_arr):
    model.fit(x_new_train,y_new_train)
    y_df['{}'.format(str(i))] = np.expm1(model.predict(x_new_test))
y_df.head()
y_ = y_df['0']/10 + y_df['1']/10 + y_df['2']/10 + y_df['3']*1.5/10 + y_df['4']*1.5/10 + y_df['5']/5 + y_final / 5
y_.iloc[:,0].head()
# save y_final to csv file
submission = pd.DataFrame()
submission['Id'] = test_ID
submission['SalePrice'] = y_.iloc[:,0]
print(submission.head())
# to_csv, and ignore the index
submission.to_csv('submission.csv', index=False)
print('Final result has been submitted')