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

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# Basic Layout details of Visualizations
large = 22; med = 16; small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (15, 10),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': small,
          'ytick.labelsize': small,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("white")
# Read House Price Train Datasets
df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
# Let's check the shape and data types in train set
print('Train set {} & Test set {}'.format(df_train.shape, df_test.shape))
# Check the details about datatype of the fields in dataset
# Train set
df_train.info()
df_test.info()
# Dropping the "Id" from train and test set. 
df_train.drop('Id', axis=1, inplace=True)
df_test.drop('Id', axis=1, inplace=True)

# Saving the target values in "y". 
y = df_train['SalePrice'].reset_index(drop=True)

# Concatinate both the datasets
df_data = pd.concat([df_train, df_test], sort=False).reset_index(drop=True)
# Drop Sale Price from df_data 
df_data.drop('SalePrice', axis=1, inplace=True)
# Check updated shape of the dataset
df_data.shape
# Update the datatypes of identified fields
df_data['MSSubClass'] = df_data['MSSubClass'].apply(str)
df_data['YrSold'] = df_data['YrSold'].astype(str)
df_data['MoSold'] = df_data['MoSold'].astype(str)
df_detail = df_data.describe(include='all')
df_detail
# Sale Price
sns.distplot(df_train['SalePrice'], bins=30)
plt.title('Distribution of Sale Price of house')

plt.show()
# Check Skewness & Kurtosis
print('Skewness of Sale Price is {}'.format(df_train['SalePrice'].skew()))
print('Kurtosis of Sale Price is {}'.format(df_train['SalePrice'].kurt()))
# Visualize correlation on Heat Map
sns.heatmap(df_train.corr(), cmap='Blues', square=True)

plt.show()
# Get top 10 Correlated fields
df_train.corr()['SalePrice'].sort_values(ascending=False)[:10]
# Let us visualize them together with the help of pair plot.
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd']

sns.pairplot(df_train[cols], kind='scatter')

plt.show()
sns.boxplot(df_train['OverallQual'], df_train['SalePrice'])
plt.title('Overall Quality of house vs Sale Price')

plt.show()
sns.boxplot(df_train['MSSubClass'], df_train['SalePrice'])
plt.title('MS Sub-Class VS Sale Price of house')

plt.show()
# Plotting Neighborhood vs Sale Price
sns.boxplot(df_train['Neighborhood'], df_train['SalePrice'])
plt.title('Neighborhood VS Sale Price of house')
plt.xticks(rotation=90)

plt.show()
# Plotting Year Built vs Sale Price
sns.boxplot(df_train['YearBuilt'], df_train['SalePrice'])
plt.title('Year Built and Sale Price of the houses')
plt.xticks(rotation=90)

plt.show()
# Visulaize missing data in combined set with the help of heatmap
sns.heatmap(df_data.isnull(), cmap='Greens')

plt.show()
# Missing values percentage
def missing_info(df):
    # Calculate total count
    total = df.isnull().sum().sort_values(ascending=False)[df.isnull().sum()!= 0]
    
    # Calculate Percent
    percent = round(df.isnull().sum().sort_values(ascending=False)/len(df)*100,2)[df.isnull().sum()!=0]
    
    return pd.concat([total, percent], axis=1, keys=['Total','Percent'])
missing_info(df_data)
# Let's plot these missing values(%) vs column_names
missing_values_count = (df_train.isnull().sum() / df_train.isnull().count()*100).sort_values(ascending=False)

base_color = sns.color_palette()[0]
plt.xlabel('Features')
plt.ylabel('Percent of missing values')
plt.title('Percent missing data by feature')

ax = sns.barplot(missing_values_count[:10].index.values, missing_values_count[:10], color = base_color)

# Adding annotations 
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate('{}%'.format(round(height,2)), (x + width/3, y + height + 0.5))

plt.show()
# Drop the columns
df_data.drop(['PoolQC','MiscFeature','Alley','Fence'], axis=1, inplace=True)
# FireplaceQu
# The houses with no fireplaces have null values in Fireplace Quality. So, we will replace them by 'NA'.
df_data['FireplaceQu'].fillna('NA', inplace=True)

# LotFrontage
# Replaced all missing values in LotFrontage by imputing the median value of each neighborhood.
df_data['LotFrontage'] = df_data.groupby('Neighborhood')['LotFrontage'].transform( lambda x: x.fillna(x.mean()))

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
# Check if have filled all the missing values
missing_info(df_data)
# Let's first get copy of this data in csv file..
df_data.to_csv('Clean_all_data.csv')
# Checking for fields with 95% similar data
common_data_col = []
for col in df_data.columns:
    pct = (df_data[col].value_counts() / len(df_data.index)).iloc[0]
    
    if pct >= 0.95:
        common_data_col.append(col)
        print('Feature {0} : {1:.2f}% '.format(col, pct*100))
# Dropping above fields
df_data.drop(['Street', 'Utilities', 'PoolArea'], axis=1, inplace=True)
df_data.columns
sns.boxplot(df_train['SalePrice'])
plt.title('Sale Price Box Plot')

plt.show()
df_train[df_train['SalePrice']>700000]
sns.distplot(np.log1p(df_train["SalePrice"]))
plt.show()
df_train['SalePrice'] = np.log1p(df_train['SalePrice'])
# Initialize list for field names
num_data = []
cat_data = []
     
# Divide list in 2 lists
for col in df_data.columns:
    if df_data[col].dtype is pd.np.dtype(object):
        cat_data.append(col)
    else:
        num_data.append(col)

print('Numeric Fields: {}'.format(num_data))
print('\nCategorical Fields: {}'.format(cat_data))
# Skewness & Kurtosis of Numeric Fields in data
def normality_check(df, data_list):
    skewness = []
    kurtosis = []
    
    for col in data_list:
        # Calculate Skewness
        skewness.append(df[col].skew()) 
    
        # Calculate Kurtosis
        kurtosis.append(df[col].kurt())
    
    return pd.DataFrame(list(zip(skewness,kurtosis)), index=data_list, columns=['Skewness', 'Kurtosis'])
# Check Skewness and Kurtosis
normality_check(df_data,num_data)
# Function to fix skewness of data
def skew_fix(df, data_list):
    "Fix skewness of Numeric Columns"
    "Accepts Data Frame and numeric column list"
    
    # Check skewness > 0.5 and transform with optimal Box-Cox transform parameter
    for col in data_list:
        if abs(df[col].skew()) > 0.5:
            df[col] = boxcox1p(df[col], boxcox_normmax(df[col] + 1))
# Fix the Skewness of data
skew_fix(df_data, num_data)
# Summing up some of the numeric values
df_data['TotalSF'] = df_data['TotalBsmtSF'] + df_data['1stFlrSF'] + df_data['2ndFlrSF']

df_data['YrBltAndRemod']= df_data['YearBuilt'] + df_data['YearRemodAdd']

df_data['Total_sqrft'] = (df_data['BsmtFinSF1'] + df_data['BsmtFinSF2'] +
                                 df_data['1stFlrSF'] + df_data['2ndFlrSF'])

df_data['Total_Bath'] = (df_data['FullBath'] + (0.5 * df_data['HalfBath']) +
                               df_data['BsmtFullBath'] + (0.5 * df_data['BsmtHalfBath']))

df_data['Total_porchsf'] = (df_data['OpenPorchSF'] + df_data['EnclosedPorch'] + df_data['ScreenPorch'] +
                              df_data['WoodDeckSF'])
# Considering Amenities of house
df_data['has2ndfloor'] = df_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
df_data['hasgarage'] = df_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
df_data['hasbsmt'] = df_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
df_data['hasfireplace'] = df_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
df_data.shape
all_data = pd.get_dummies(df_data).reset_index(drop=True)
all_data.shape
y = df_train['SalePrice']

X = all_data.iloc[:len(y), :]

org_test = all_data.iloc[len(y):, :]

print(X.shape, y.shape, org_test.shape)
