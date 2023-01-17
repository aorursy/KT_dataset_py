# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Data Visualization

import seaborn as sns

import matplotlib.pyplot as plt 



# Stats

from scipy.stats import skew, norm

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax



# Preprocessing

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold



# Models

from sklearn.linear_model import LinearRegression



# Misc

from sklearn import metrics



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Read the train & test dataset

df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
# Check data in both files

df_train.head()
# Test Data

df_test.head()
# Check shape of the DataFrame

print('Size of train data', df_train.shape)

print('Size of test data', df_test.shape)
# Check data type of all the features 

df_train.info()
# Let us take backup of each file before starting with out analysis

df_train_bkp = df_train.copy()

df_test_bkp = df_test.copy()
# Finding missing values 

def missing_info(df):

    # Calculate total count

    total = df.isnull().sum().sort_values(ascending=False)[df.isnull().sum()!= 0]

    

    # Calculate Percent

    percent = round(df.isnull().sum().sort_values(ascending=False)/len(df)*100,2)[df.isnull().sum()!=0]

    

    return pd.concat([total, percent], axis=1, keys=['Total','Percent'])
# Missing information from Train Set

missing_info(df_train)
# Missing information from Test Set

missing_info(df_test)
# Visualize the top 10 columns which are missing data

missing_values_train = (df_train.isnull().sum() / df_train.isnull().count()*100).sort_values(ascending=False)



fig = plt.figure(figsize=(15,10))



base_color = sns.color_palette()[0]



sns.barplot(missing_values_train[:10].index.values, missing_values_train[:10], color = base_color)

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Percent missing data for Train Set', fontsize=15)



plt.show()
# Check the distribution of House Price ranges

fig = plt.figure(figsize=(15,10))



sns.distplot(df_train['SalePrice'], bins=30)

plt.title('Distribution of Sale Price of house')



plt.show()
fig = plt.figure(figsize=(15,10))



sns.heatmap(df_train.corr(), cmap='Blues', square=True)



plt.show()
# Let's check top 15 correlating features

df_train.corr()['SalePrice'].sort_values(ascending=False)[:15]
# Let us visualize them together with the help of pair plot.

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd']



sns.pairplot(df_train[cols], kind='scatter')



plt.show()
# Year Built

fig = plt.figure(figsize=(15,10))



sns.boxplot(df_train['YearBuilt'], df_train['SalePrice'])

plt.xticks(rotation=90)



plt.show()
# MasVnrArea

fig = plt.figure(figsize=(15,10))



sns.scatterplot(df_train['MasVnrArea'], df_train['SalePrice'])



plt.show()
# House Style

fig = plt.figure(figsize=(15,10))



sns.countplot(df_train['HouseStyle'], order=df_train['HouseStyle'].value_counts().index)



plt.show()
# Neighborhood

fig = plt.figure(figsize=(15,10))



sns.boxplot(df_train['Neighborhood'], df_train['SalePrice'])

plt.xticks(rotation=90)



plt.show()
## Dropping the "Id" from train and test set. 

df_train.drop('Id', axis=1, inplace=True)

df_test.drop('Id', axis=1, inplace=True)



## Saving the target values in "y_train". 

y = df_train['SalePrice'].reset_index(drop=True)
# Concatinate both DataFrames

df_data = pd.concat([df_train, df_test], sort=False).reset_index(drop=True)
df_data.head()
df_data.tail()
# Drop Target columns SalePrice from concatenated DataFrame

df_data.drop('SalePrice', axis=1, inplace=True)
# Let us again check combined missing values:

missing_info(df_data)
# Let us drop the columns which are having more than 80% missing data

df_data.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis=1, inplace=True)
# Convert the data type to string for following columns:



df_data['MSSubClass'] = df_data['MSSubClass'].apply(str)

df_data['YrSold'] = df_data['YrSold'].astype(str)

df_data['MoSold'] = df_data['MoSold'].astype(str)
# FireplaceQu

# The houses with no fireplaces have null values in Fireplace Quality. So, we will replace them by 'NA'.

df_data['FireplaceQu'].fillna('NA', inplace=True)



# LotFrontage

# The NaN values actually represents that there is no street connected to the property. Replace them with 0

df_data['LotFrontage'].fillna(0, inplace=True)



# Garage

# From above report, the count for no garage house is zero i.e. null values. Replace these null values as 'NA'.

df_data.fillna({'GarageFinish':'NA', 'GarageQual':'NA', 'GarageCond':'NA', 'GarageType':'NA', 'GarageYrBlt':0, 'GarageCars':0, 'GarageArea':0}, inplace=True)



# Basement

# If the value is Nan, there is no basment and hence will replace it with 'NA'

df_data.fillna({'BsmtQual':'NA', 'BsmtCond':'NA', 'BsmtExposure':'NA', 'BsmtFinType1':'NA', 'BsmtFinType2':'NA', }, inplace=True)

df_data.fillna({'BsmtFinSF1':0, 'BsmtFinSF2':0, 'BsmtUnfSF':0, 'TotalBsmtSF':0, 'BsmtFullBath':0, 'BsmtHalfBath':0}, inplace=True)



# MasVnrType & MasVnrArea

# Replace Null values for 'MasVnrType' with 'None' and Area with 0.

df_data.fillna({'MasVnrType':'None', 'MasVnrArea':0}, inplace=True)



# MSZoning

# Replace null values with the most common values

df_data['MSZoning'].fillna(df_data['MSZoning'].mode()[0], inplace=True)



# Utilities

# Filling null values with most common Utilities

df_data['Utilities'].fillna(df_data['Utilities'].mode()[0], inplace=True)



# Functional

df_data['Functional'].fillna(df_data['Functional'].mode()[0], inplace=True)



# Kitchen Quality

df_data['KitchenQual'].fillna(df_data['KitchenQual'].mode()[0], inplace=True)



# Exterior

# Replacing null values with most common Exterior

df_data['Exterior1st'].fillna(df_data['Exterior1st'].mode()[0], inplace=True)

df_data['Exterior2nd'].fillna(df_data['Exterior2nd'].mode()[0], inplace=True)



# Sale Type

# Fill the null values with most common Sale Type

df_data['SaleType'].fillna(df_data['SaleType'].mode()[0], inplace=True)



# Electrical

df_data['Electrical'].fillna(df_data['Electrical'].mode()[0], inplace=True)
# Check missing values status again

missing_info(df_data)
# Let's go back to the Target field and fix the skewness, but let's visualize first

fig = plt.figure(figsize=(10,5))



sns.distplot(df_train['SalePrice'], bins=30)



plt.show()
df_train["SalePrice"] = np.log1p(df_train["SalePrice"])



# Plot target variable again

fig = plt.figure(figsize=(15,10))



sns.distplot(df_train['SalePrice'], bins=30)



plt.show()
num_col = df_data.dtypes[df_data.dtypes != "object"].index



num_col_skewness = df_data[num_col].apply(lambda x: skew(x)).sort_values(ascending=False)



num_col_skewness
# Fixing the skewness using box-cox transaformation

def fix_skewness(df):

    "This function will take input as DataFrame and fix skewness of all the numeric fields using box-cox transaformation"

    

    num_col = df.dtypes[df.dtypes != "object"].index



    num_col_skewness = df[num_col].apply(lambda x: skew(x)).sort_values(ascending=False)

    

    high_skew = num_col_skewness[abs(num_col_skewness) > 0.5]

    skewed_cols = high_skew.index

    

    for col in skewed_cols:

        df[col] = boxcox1p(df[col], boxcox_normmax(df[col] + 1))
fix_skewness(df_data)
# Let us check one of numeric columns like "GrLivArea"

fig = plt.figure(figsize=(10,10))



sns.distplot(df_data['GrLivArea'], bins=30)



plt.show()
# Considering Numeric Columns

# Total Surface Area 

df_data['TotalSF'] = df_data['TotalBsmtSF'] + df_data['1stFlrSF'] + df_data['2ndFlrSF']



# Adding the Year Built and Year Remodelled

df_data['YrBltAndRemod'] = df_data['YearBuilt'] + df_data['YearRemodAdd']



# Total Surface Area of house

df_data['Total_sqr_footage'] = (df_data['BsmtFinSF1'] + df_data['BsmtFinSF2'] +

                                 df_data['1stFlrSF'] + df_data['2ndFlrSF'])



# Total Number of Bathrooms in a house

df_data['Total_Bathrooms'] = (df_data['FullBath'] + (0.5 * df_data['HalfBath']) +

                               df_data['BsmtFullBath'] + (0.5 * df_data['BsmtHalfBath']))



# Total Porch Area

df_data['Total_porch_sf'] = (df_data['OpenPorchSF'] + df_data['3SsnPorch'] +

                              df_data['EnclosedPorch'] + df_data['ScreenPorch'] +

                              df_data['WoodDeckSF'])



# Considering Amenities of house

df_data['haspool'] = df_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

df_data['has2ndfloor'] = df_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

df_data['hasgarage'] = df_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

df_data['hasbsmt'] = df_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

df_data['hasfireplace'] = df_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
# Let us check how many columns have more than 90% same data:

def common_data(df):

    for col in df.columns:

        pct = (df[col].value_counts() / len(df.index)).iloc[0]

    

        if pct >= 0.90:

            print('Feature {0} : {1:.2f}% similar data'.format(col, pct*100))
common_data(df_data)
# Drop 'Street', 'Utilities' & 'PoolArea' columns

df_data.drop(['Street', 'Utilities'], axis=1, inplace=True)
# Check shape of DataFrame

df_data.shape
df_all_data = pd.get_dummies(df_data).reset_index(drop=True)

df_all_data.shape
X = df_all_data.iloc[:len(y), :]



X_dftest = df_all_data.iloc[len(y):, :]

print(X.shape, y.shape, X_dftest.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
lm = LinearRegression(normalize=True, n_jobs=-1)
# Fit the model

lm.fit(X_train, y_train)
# Predict the output for X_test

y_pred = lm.predict(X_test)
# Mean Squared Error

print ('%.2f'%metrics.mean_squared_error(y_test, y_pred))
# R2

print(metrics.r2_score(y_test, y_pred))
# Using Cross Validation

lm1 = LinearRegression()



cv = KFold(shuffle=True, random_state=2, n_splits=10)



scores = cross_val_score(lm1, X, y, cv = cv, scoring = 'neg_mean_absolute_error')
print(scores)
# get mean of these scores

print ('%.8f'%scores.mean())
# Read Submission file

print('Predict submission')



submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
submission.head()
X_dftest.head()
submission.iloc[:,1] = np.floor(lm.predict(X_dftest))
submission.head()
submission.to_csv("submission.csv", index=False)
submission.to_csv(r"C:\Users\mdesh\OneDrive\Desktop\Mayuree\Study\Machine Learning Practise\House Price Prediction\submission.csv", index=False)