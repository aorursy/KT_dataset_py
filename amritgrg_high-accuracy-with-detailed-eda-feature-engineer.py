# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from scipy.stats import norm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
# lets load data
train_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
# create copy of above dataframe. 
train = train_df.copy()
test = test_df.copy()
# lets see shape of datas
print('train data shape: ', train.shape)
print('test data shape: ', test.shape)
# lets view first five records in train data
train.head()
# lets view first 5 observations in test data
test.head()
# you can see null values even in firstt 5 observations as seen above
# lets find the null values in data

total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
# Id attributes have no special meaning in in regression so lets drop them
train.drop('Id', axis=1, inplace = True)
test.drop('Id', axis=1, inplace=True)
## plotting distribution of target feature
sns.distplot(train['SalePrice'])
plt.show()
# Numerical features
Numerical_feat = [feature for feature in train.columns if train[feature].dtypes != 'O']
print('Total numerical features: ', len(Numerical_feat))
print('\nNumerical Features: ', Numerical_feat)
# making a glance of first 5 observations
train[Numerical_feat].head()
# Zoomed heatmap, correlation matrix
sns.set(rc={'figure.figsize':(12,8)})
correlation_matrix = train.corr()

k = 10             #number of variables for heatmap
cols = correlation_matrix.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
## these are selected features from heatmap
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

# Discrete features

discrete_feat = [feature for feature in Numerical_feat if len(train[feature].unique())<25]
print('Total discrete features: ', len(discrete_feat))
print('\n', discrete_feat)
# glancing first five records of discrete features
train[discrete_feat].head()
train[discrete_feat].info()
# Lets find unique values in each discrete features
for feature in discrete_feat:
    print('Uique values of ', feature, ':')
    print(train[feature].unique())
    print('\n')
    
## Lets Find the realtionship between discrete features and SalePrice

#plt.figure(figsize=(8,6))

for feature in discrete_feat:
    data=train.copy()
    plt.figure(figsize=(8,6))
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()
continuous_features = [feature for feature in Numerical_feat if feature not in discrete_feat]
print('The numbers of continuous features: ', len(continuous_features))
print('\n', continuous_features)
## Lets analyse the continuous values by creating histograms to understand the distribution

train[continuous_features].hist(bins=25)
plt.show()
## let us now examine the relationship between continuous features and SalePrice
## Before that lets find continous features that donot contain zero values

continuous_nozero = [feature for feature in continuous_features if 0 not in data[feature].unique() and feature not in ['YearBuilt', 'YearRemodAdd']]

for feature in continuous_nozero:
    plt.figure(figsize=(8,6))
    data = train.copy()
    data[feature] = np.log(data[feature])
    data['SalePrice'] = np.log(data['SalePrice'])
    plt.scatter(data[feature], data['SalePrice'])
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.show()
## Normality and distribution checking for continous features
for feature in continuous_nozero:
    plt.figure(figsize=(6,6))
    data = train.copy()
    sns.distplot(data[feature])
    plt.show()
# categorical features
categorical_feat = [feature for feature in train.columns if train[feature].dtypes=='O']
print('Total categorical features: ', len(categorical_feat))
print('\n',categorical_feat)
# lets view few samples 
train[categorical_feat].head()
# lets find unique values in each categorical features
for feature in categorical_feat:
    print('{} has {} categories. They are:'.format(feature,len(train[feature].unique())))
    print(train[feature].unique())
    print('\n')

# let us find relationship of categorical with target variable

for feature in categorical_feat:
    data=train_df.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()

Train = train_df.shape[0]
Test = test_df.shape[0]
target_feature = train_df.SalePrice.values
combined_data = pd.concat((train_df, test_df)).reset_index(drop=True)
combined_data.drop(['SalePrice','Id'], axis=1, inplace=True)
print("all_data size is : {}".format(combined_data.shape))
# let's find the missing data in combined dataset

total = combined_data.isna().sum().sort_values(ascending=False)
percent = (combined_data.isnull().sum()/combined_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
# Lets first handle numerical features will nan value
numerical_nan = [feature for feature in combined_data.columns if combined_data[feature].isna().sum()>1 and combined_data[feature].dtypes!='O']
numerical_nan
combined_data[numerical_nan].isna().sum()
## Replacing the numerical Missing Values

for feature in numerical_nan:
    ## We will replace by using median since there are outliers
    median_value=combined_data[feature].median()
    
    combined_data[feature].fillna(median_value,inplace=True)
    
combined_data[numerical_nan].isnull().sum()
# categorical features with missing values
categorical_nan = [feature for feature in combined_data.columns if combined_data[feature].isna().sum()>1 and combined_data[feature].dtypes=='O']
print(categorical_nan)
combined_data[categorical_nan].isna().sum()
# replacing missing values in categorical features
for feature in categorical_nan:
    combined_data[feature] = combined_data[feature].fillna('None')
combined_data[categorical_nan].isna().sum()

# these are selected features from EDA section
features = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
# plot bivariate distribution (above given features with saleprice(target feature))
for feature in features:
    if feature!='SalePrice':
        plt.scatter(train_df[feature], train_df['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.show()
#Deleting outliers for GrLivArea
train_df = train_df.drop(train_df[(train_df['GrLivArea']>4000) & (train_df['SalePrice']<300000)].index)

plt.scatter(train_df['GrLivArea'], train_df['SalePrice'])
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
plt.show()
#Deleting outliers for TotalBsmtSF
train_df = train_df.drop(train_df[(train_df['TotalBsmtSF']>5000) & (train_df['SalePrice']<300000)].index)

plt.scatter(train_df['TotalBsmtSF'],train_df['SalePrice'])
plt.xlabel('TotalBsmtSF')
plt.ylabel('SalePrice')
plt.show()
# these are selected features from EDA section
features = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

# selecting continuous features from above
continuous_features = ['SalePrice', 'GrLivArea', 'TotalBsmtSF']
# checking distribution of continuous features(histogram plot)
for feature in continuous_features:
    if feature!='SalePrice':
        sns.distplot(combined_data[feature], fit=norm)
        plt.show()
    else:
        sns.distplot(train_df['SalePrice'], fit=norm)
        plt.show()
#create column for new variable (one is enough because it's a binary categorical feature)
#if area>0 it gets 1, for area==0 it gets 0
# This idea is from Pedro Marcelino, PhD notebook.

combined_data['HasBsmt'] = 0  # at first o for all observations in 'HasBsmt'
combined_data.loc[combined_data['TotalBsmtSF']>0,'HasBsmt'] = 1  # assign 1 for those with no basement 
#transform data
combined_data.loc[combined_data['HasBsmt']==1,'TotalBsmtSF'] = np.log(combined_data['TotalBsmtSF'])
combined_data['GrLivArea'] = np.log(combined_data['GrLivArea'])
train_df['SalePrice'] = np.log(train_df['SalePrice'])
# we have log transormed above skewed data. Now lets see their distribution
for feature in continuous_features:
    if feature!='SalePrice':
        sns.distplot(combined_data[feature], fit=norm)
        plt.show()
    else:
        sns.distplot(train_df['SalePrice'],fit=norm)
        plt.show()

## these are features that seems to give information in order form
## taken from Serigne's notebook

ordinal_features = ['FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
                 'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
                 'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
                 'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
                 'YrSold', 'MoSold']
print(len(ordinal_features))
## Credit for Serigne 

#MSSubClass=The building class
combined_data['MSSubClass'] = combined_data['MSSubClass'].apply(str)


#Changing OverallCond into a categorical variable
combined_data['OverallCond'] = combined_data['OverallCond'].astype(str)


#Year and month sold are transformed into categorical features.
#combined_data['YrSold'] = combined_data['YrSold'].astype(str)
#combined_data['MoSold'] = combined_data['MoSold'].astype(str)

combined_data[ordinal_features].head(10)
# so let's label encode above ordinal features
from sklearn.preprocessing import LabelEncoder
for feature in ordinal_features:
    encoder = LabelEncoder()
    combined_data[feature] = encoder.fit_transform(list(combined_data[feature].values))
# Now lets see label encoded data
combined_data[ordinal_features].head()
## One hot encoding or getting dummies 

dummy_ordinals = pd.get_dummies(ordinal_features) 
dummy_ordinals.head()
# creating dummy variables

combined_data = pd.get_dummies(combined_data)
print(combined_data.shape)
combined_data.head(10)
# let's first see descriptive stat info 
combined_data.describe()
## we willtake all features from combined_dummy_data 
features_to_scale = [feature for feature in combined_data]
print(len(features_to_scale))
## Now here is where we will scale our data using sklearn module.

from sklearn.preprocessing import MinMaxScaler

cols = combined_data.columns  # columns of combined_dummy_data

scaler = MinMaxScaler()
combined_data = scaler.fit_transform(combined_data[features_to_scale])
# after scaling combined_data it is now in ndarray datypes
# so we will create DataFrame from it
combined_scaled_data = pd.DataFrame(combined_data, columns=[cols])
combined_scaled_data.head() # this is the same combined_dummy_data in scaled form.
# lets see descriptive stat info 
combined_scaled_data.describe()
train_df.shape, test_df.shape, combined_scaled_data.shape, combined_data.shape
# separate train data and test data 
train_data = combined_scaled_data.iloc[:1460,:]
test_data = combined_scaled_data.iloc[1460:,:]

train_data.shape, test_data.shape
## lets add target feature to train_data
train_data['SalePrice']= train_df['SalePrice']  # This saleprice is normalized. Its very impportant
train_data = train_data
train_data.head(10)
test_data = test_data.reset_index()
test_data.tail(10)

## ugh.. it seems outliers that we droped earlier haven't droped from combined data.
## that makes sense since we had droped only from train data before not from combined data.
## S0 we will drop them here

#Deleting outliers for TotalBsmtSF
#train_data = train_data.drop(train_data[(train_data['TotalBsmtSF']>5000) & (train_data['SalePrice']<300000)].index)

#Deleting outliers for GrLivArea
#train_data = train_data.drop(train_data[(train_data['GrLivArea']>4000) & (train_data['SalePrice']<300000)].index)


dataset = train_data.copy()  # copy train_data to dataset variable
dataset.head()
dataset = dataset.dropna()
## lets create dependent and target feature vectors

X = dataset.drop(['SalePrice'],axis=1)
Y = dataset[['SalePrice']]

X.shape, Y.shape
Y.head()
# lets do feature selection here

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

# define feature selection
fs = SelectKBest(score_func=f_regression, k=27)
# apply feature selection
X_selected = fs.fit_transform(X, Y)
print(X_selected.shape)

cols = list(range(1,28))

## create dataframe of selected features

selected_feat = pd.DataFrame(data=X_selected,columns=[cols])
selected_feat.head()

# perform train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(selected_feat,Y,test_size=0.3,random_state=0)
x_train.shape, x_test.shape
from sklearn.linear_model import LinearRegression
from sklearn import metrics

lr = LinearRegression()
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test) # predicting test data
y_pred[:10]
# Evaluating the model
print('R squared score',metrics.r2_score(y_test,y_pred))

print('\nMean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# check for underfitting and overfitting
print('Train Score: ', lr.score(x_train,y_train))
print('Test Score: ', lr.score(x_test,y_test))
## scatter plot of original and predicted target test data
plt.figure(figsize=(8,6))
plt.scatter(y_test,y_pred)
plt.xlabel('y_tes')
plt.ylabel('y_pred')
plt.show()
## Lets do error plot
## to get error in prediction just substract predicted values from original values

error = list(y_test.values-y_pred)
plt.figure(figsize=(8,6))
sns.distplot(error)
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=100)
rf_reg.fit(x_train,y_train)
y_pred = rf_reg.predict(x_test)
print(y_pred[:10])
## evaluating the model

print('R squared error',metrics.r2_score(y_test,y_pred))

print('\nMean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# check score
print('Train Score: ', rf_reg.score(x_train,y_train))
print('Test Score: ', rf_reg.score(x_test,y_test))
## scatter plot of original and predicted target test data
plt.figure(figsize=(8,6))
plt.scatter(y_test,y_pred)
plt.xlabel('y_tes')
plt.ylabel('y_pred')
plt.show()
## Lets do error plot
## to get error in prediction just substract predicted values from original values

error = list(y_test.values-y_pred)
plt.figure(figsize=(8,6))
sns.distplot(error)
