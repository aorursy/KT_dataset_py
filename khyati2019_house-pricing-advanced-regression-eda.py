# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Importing all required libraries

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import linear_model

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.model_selection import GridSearchCV



#Scaling libraries

from sklearn .preprocessing import scale



import os

import datetime





# hide warnings

import warnings

warnings.filterwarnings('ignore')



#To display all columns and rows

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)
# Reading the dataset

housing = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')



housing.head()
# summary of the dataset

print(housing.info())
# Checking the percentage of missing values

round(100*(housing.isnull().sum()/len(housing.index)), 2)
#Dropping null values for the column 'GarageYrBlt'

housing = housing.dropna(axis=0, subset=['GarageYrBlt'])

housing['GarageYrBlt'].isnull().sum()
#Converting GarageYrBlt from float to int and converting to datetime

housing['GarageYrBlt']= housing['GarageYrBlt'].astype(int)



# Converting year columns to datetime

housing['GarageYrBlt'] = pd.to_datetime(housing['GarageYrBlt'].astype(str), format='%Y')

#housing['GarageYrBlt'] = pd.to_datetime(housing['GarageYrBlt'], unit='s')

housing['YearRemodAdd'] = pd.to_datetime(housing['YearRemodAdd'].astype(str), format='%Y')

housing['YrSold'] = pd.to_datetime(housing['YrSold'].astype(str), format='%Y')

housing['YearBuilt'] = pd.to_datetime(housing['YearBuilt'].astype(str), format='%Y')





# Converting the Year columns from datetime to date

housing['GarageYrBlt'] = housing['GarageYrBlt'].dt.date

housing['YearRemodAdd'] = housing['YearRemodAdd'].dt.date

housing['YrSold'] = housing['YrSold'].dt.date

housing['YearBuilt'] = housing['YearBuilt'].dt.date
# Calcualting the age using the Year column and today's date



now = datetime.date.today()

housing['GarageYrBltAge_in_years'] = now - housing['GarageYrBlt']

housing['YearRemodAddAge_in_years'] = (now - housing['YearRemodAdd'])/365

housing['YrSoldAge_in_years'] = (now - housing['YrSold'])/365

housing['YearBuiltAge_in_years'] = (now - housing['YearBuilt'])/365
# Convert age to int

housing['GarageYrBltAge_in_years']= housing.apply(lambda row: row.GarageYrBltAge_in_years.days, axis=1)



housing['YearRemodAddAge_in_years']= housing.apply(lambda row: row.YearRemodAddAge_in_years.days, axis=1)

housing['YrSoldAge_in_years']= housing.apply(lambda row: row.YrSoldAge_in_years.days, axis=1)

housing['YearBuiltAge_in_years']= housing.apply(lambda row: row.YearBuiltAge_in_years.days, axis=1)

# we can drop the original columns of years 

#housing = housing.drop(['GarageYrBlt','YearRemodAdd','YrSold','YearBuilt'],1)

housing.info()
# plot per sale price and YearBuild

plt.scatter(housing['YearBuiltAge_in_years'],housing['SalePrice'])

plt.ylabel('Sale Price')

plt.xlabel('YearBuiltAge_in_years')
# plot per sale price and YearBuild

plt.scatter(housing['YearRemodAddAge_in_years'],housing['SalePrice'])

plt.ylabel('Sale Price')

plt.xlabel('YearRemodAddAge_in_years')
# plot per sale price and YearBuild

plt.scatter(housing['OverallQual'],housing['SalePrice'])

plt.ylabel('Sale Price')

plt.xlabel('OverallQual')
# all numeric (float and int) variables in the dataset

housing_numeric = housing.select_dtypes(include=['float64', 'int64'])

housing_numeric.head()

# plotting pairplot with few numeric variables.

housing_numeric_plot = housing[['SalePrice','GarageYrBltAge_in_years','YearRemodAddAge_in_years','OverallQual','YearBuiltAge_in_years','LotArea','GarageCars','YrSoldAge_in_years','MoSold']]

# pairwise scatter plot



plt.figure(figsize=(20, 10))

sns.pairplot(housing_numeric_plot)

plt.show()
housing.head()
# Checking the percentage of missing values

round(100*(housing.isnull().sum()/len(housing.index)), 2)
# Dropping columns having more tha 80% null values

housing_new = housing

housing_new = housing_new.drop(['Alley','PoolQC','Fence','MiscFeature'],1)



# Checking the percentage of missing values

round(100*(housing_new.isnull().sum()/len(housing_new.index)), 2)
#'FireplaceQu' has max nulls, checking values to check if they can be replaced 

housing_new['FireplaceQu'].astype('category').value_counts()

#### we would impute the Null fields with values 'NA - as No Fireplace'

housing_new['FireplaceQu'].describe()



# Replace null values with 'No Info'

housing_new['FireplaceQu'] = housing_new['FireplaceQu'].replace(np.nan, 'No Info')


# Checking the percentage of missing values

round(100*(housing_new.isnull().sum()/len(housing_new.index)), 2)
# LotFrontage: Linear feet of street connected to property -- null values is 18.27%

housing_new['LotFrontage'].describe()
#imputing null values for LotFrontage with mean values

housing_new['LotFrontage'] = housing_new['LotFrontage'].replace(np.nan, housing_new['LotFrontage'].mean())
# Data Cleaning of other columns having null values:



#MasVnrType: Masonry veneer type



#       BrkCmn	Brick Common

#       BrkFace	Brick Face

#       CBlock	Cinder Block

#       None	None

#       Stone	Stone

housing_new['MasVnrType'].describe()



#Replacing null values with "No Info"

housing_new['MasVnrType'] = housing_new['MasVnrType'].replace(np.nan, 'No Info')



# MasVnrArea: Masonry veneer area in square feet

housing_new['MasVnrArea'].describe()



#imputing null values for MasVnrArea with mean values

housing_new['MasVnrArea'] = housing_new['MasVnrArea'].replace(np.nan, housing_new['MasVnrArea'].mean())



#BsmtQual: Evaluates the height of the basement



#       Ex	Excellent (100+ inches)	

#       Gd	Good (90-99 inches)

#       TA	Typical (80-89 inches)

#       Fa	Fair (70-79 inches)

#       Po	Poor (<70 inches

#       NA	No Basement

housing_new['BsmtQual'].describe()



#Replacing null values with "No Info"

housing_new['BsmtQual'] = housing_new['BsmtQual'].replace(np.nan, 'No Info')



#BsmtCond: Evaluates the general condition of the basement



#       Ex	Excellent

#       Gd	Good

#       TA	Typical - slight dampness allowed

#       Fa	Fair - dampness or some cracking or settling

#       Po	Poor - Severe cracking, settling, or wetness

#       NA	No Basement

housing_new['BsmtCond'].describe()



#Replacing null values with "No Info"

housing_new['BsmtCond'] = housing_new['BsmtCond'].replace(np.nan, 'No Info')



housing_new['BsmtExposure'].describe()



#Replacing null values with "No Info"

housing_new['BsmtExposure'] = housing_new['BsmtExposure'].replace(np.nan, 'No Info')





housing_new['BsmtFinType1'].describe()



#Replacing null values with "No Info"

housing_new['BsmtFinType1'] = housing_new['BsmtFinType1'].replace(np.nan, 'No Info')

housing_new['BsmtFinType2'] = housing_new['BsmtFinType2'].replace(np.nan, 'No Info')





housing_new['Electrical'].describe()



housing_new['Electrical'] = housing_new['Electrical'].replace(np.nan, 'No Info')


# Checking the percentage of missing values

round(100*(housing_new.isnull().sum()/len(housing_new.index)), 2)
# Let us take the numerical columns to check the correlation between them



# all numeric (float and int) variables in the dataset

housing_new_numeric1 = housing_new.select_dtypes(include=['float64', 'int64'])

housing_new_numeric1.head()
# correlation matrix

cor = housing_new_numeric1.corr()

cor
# plotting correlations on a heatmap



# figure size

plt.figure(figsize=(30,18))



# heatmap

sns.heatmap(cor, cmap="YlGnBu", annot=True)

plt.show()

housing_new_numeric1.columns
housing_new_numeric_drop = ['Id','MSSubClass','OverallCond','BsmtFinSF2','BsmtUnfSF','1stFlrSF','LowQualFinSF','BsmtHalfBath','BedroomAbvGr','KitchenAbvGr','GarageArea','3SsnPorch','ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold','YrSoldAge_in_years']



housing_new = housing_new.drop(housing_new_numeric_drop,1)



                                                
# Checking outliers at 25%, 50%, 75%, 90%, 95% and 99%

housing_new_numeric1.describe(percentiles=[.25, .5, .75, .90, .95, .99])
## Outlier treatement

plt.boxplot(housing_new.YearBuiltAge_in_years)

Q1 = housing_new.YearBuiltAge_in_years.quantile(0.05)

Q3 = housing_new.YearBuiltAge_in_years.quantile(0.95)

IQR = Q3 - Q1

housing_new = housing_new[(housing_new.YearBuiltAge_in_years>=Q1)&(housing_new.YearBuiltAge_in_years<=Q3)]
## Outlier treatement

plt.boxplot(housing_new.LotArea)

Q1 = housing_new.LotArea.quantile(0.05)

Q3 = housing_new.LotArea.quantile(0.95)

IQR = Q3 - Q1

housing_new = housing_new[(housing_new.LotArea>=Q1)&(housing_new.LotArea<=Q3)]
# all numeric (float and int) variables in the dataset

housing_new_numeric1 = housing_new.select_dtypes(include=['float64', 'int64'])

housing_new_numeric1.head()
# Checking outliers at 25%, 50%, 75%, 90%, 95% and 99%

housing_new_numeric1.describe(percentiles=[.25, .5, .75, .90, .95, .99])
housing_new.info()

#Now we have 1141 rows and 64 columns
# split into X and y

housing_new1 = housing_new 

housing_new1 = housing_new1.drop(['SalePrice'], axis=1)

X = housing_new1



X.info()
y=housing_new['SalePrice']
# creating dummy variables for categorical variables



# subset all categorical variables

housing_categorical = X.select_dtypes(include=['object'])

housing_categorical.head()

housing_categorical.info()
# convert into dummies

housing_dummies = pd.get_dummies(housing_categorical, drop_first=True)

housing_dummies.head()
# drop categorical variables 

X = X.drop(list(housing_categorical.columns), axis=1)
# concat dummy variables with X

X = pd.concat([X, housing_dummies], axis=1)

X.columns
# scaling the features

from sklearn.preprocessing import scale



# storing column names in cols, since column names are lost after 

# scaling (the df is converted to a numpy array)

cols = X.columns

X = pd.DataFrame(scale(X))

X.columns = cols

X.columns
from sklearn.preprocessing import StandardScaler



from sklearn import preprocessing

#min_max_scaler = preprocessing.MinMaxScaler()

# scale

scaler = StandardScaler()

scaler.fit(X)
