import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

import os

from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')
#Reading the file from the system
housing_df = pd.read_csv("../input/Housing Train Dataset.csv")

housing_df.head()
#Checking the shape of the dataframe
housing_df.shape
#Checking for data types
housing_df.info()
#Checking the value distribution of numerical columns
housing_df.describe()
#Checking all the columns
housing_df.columns
# Checking whether any column has only null values
housing_df.isnull().all(axis=0).any()
#Checking for columns with only 1 unique value
housing_df.loc[:,housing_df.nunique()==1].columns
# Checking the percentage of missing value
round(100*(housing_df.isnull().sum()/len(housing_df.index)), 2)
# Selecting columns having more than 80% of missing values
housing_df.loc[:,(round(100*(housing_df.isnull().sum()/len(housing_df.index)), 2)>=30)].columns
#Checking unique values in columns having nan values greater than 30% (5 colummns)
print("Alley" ,housing_df['Alley'].unique())
print("FireplaceQu" ,housing_df['FireplaceQu'].unique())
print("PoolQC" ,housing_df['PoolQC'].unique())
print("Fence" ,housing_df['Fence'].unique())
print("MiscFeature" ,housing_df['MiscFeature'].unique())
# Analyzing the columns 'Alley' which has 93% of missing values
print(housing_df['Alley'].value_counts())
#Plotting count plot for Alley column
plt.figure(figsize=(10,10))
plt.subplot(2, 2, 1)
sns.countplot(x='Alley', data=housing_df)
plt.title('Distribution of Alley before imputing')
plt.xticks(fontsize=11, rotation=90)
plt.yticks(fontsize=11, rotation=0)

#Instead of dropping the Alley column which seems of high relevancy lets impute the missing with 
#'No Alley Access' as it corresponds to nan
housing_df.loc[:,'Alley'] = housing_df.loc[:,'Alley'].replace(np.nan, 'No Alley Access')

plt.subplot(2, 2, 2)
sns.countplot(x='Alley', data=housing_df)
plt.title('Distribution of Alley after imputing')
plt.xticks(fontsize=11, rotation=90)
plt.yticks(fontsize=11, rotation=0)

plt.tight_layout()
plt.show()
# Analyzing the columns 'FireplaceQu' which has 47% of missing values
print(housing_df['FireplaceQu'].value_counts())
#Plotting count plot for FireplaceQu column
plt.figure(figsize=(10,10))
plt.subplot(2, 2, 1)
sns.countplot(x='FireplaceQu', data=housing_df)
plt.title('Distribution of FireplaceQu before imputing')
plt.xticks(fontsize=11, rotation=90)
plt.yticks(fontsize=11, rotation=0)

#Instead of dropping the FireplaceQu column which seems of high relevancy lets impute the missing with 
#'NA' as it corresponds to nan
housing_df.loc[:,'FireplaceQu'] = housing_df.loc[:,'FireplaceQu'].replace(np.nan, 'No Fireplace')

plt.subplot(2, 2, 2)
sns.countplot(x='FireplaceQu', data=housing_df)
plt.title('Distribution of FireplaceQu after imputing')
plt.xticks(fontsize=11, rotation=90)
plt.yticks(fontsize=11, rotation=0)

plt.tight_layout()
plt.show()
# Analyzing the columns 'FireplaceQu' which has 99% of missing values
print(housing_df['PoolQC'].value_counts())
#Plotting count plot for PoolQC column
plt.figure(figsize=(10,10))
plt.subplot(2, 2, 1)
sns.countplot(x='PoolQC', data=housing_df)
plt.title('Distribution of PoolQC before imputing')
plt.xticks(fontsize=11, rotation=90)
plt.yticks(fontsize=11, rotation=0)

#Instead of dropping the PoolQC column which seems of high relevancy lets impute the missing with 
#'No Pool' as it corresponds to nan
housing_df.loc[:,'PoolQC'] = housing_df.loc[:,'PoolQC'].replace(np.nan, 'No Pool')

plt.subplot(2, 2, 2)
sns.countplot(x='PoolQC', data=housing_df)
plt.title('Distribution of PoolQC after imputing')
plt.xticks(fontsize=11, rotation=90)
plt.yticks(fontsize=11, rotation=0)

plt.tight_layout()
plt.show()
# Analyzing the columns 'FireplaceQu' which has 80% of missing values
print(housing_df['Fence'].value_counts())
#Plotting count plot for Fence column
plt.figure(figsize=(10,10))
plt.subplot(2, 2, 1)
sns.countplot(x='Fence', data=housing_df)
plt.title('Distribution of Fence before imputing')
plt.xticks(fontsize=11, rotation=90)
plt.yticks(fontsize=11, rotation=0)

#Instead of dropping the Fence column which seems of high relevancy lets impute the missing with 
#'No Fence' as it corresponds to nan
housing_df.loc[:,'Fence'] = housing_df.loc[:,'Fence'].replace(np.nan, 'No Fence')

plt.subplot(2, 2, 2)
sns.countplot(x='Fence', data=housing_df)
plt.title('Distribution of Fence after imputing')
plt.xticks(fontsize=11, rotation=90)
plt.yticks(fontsize=11, rotation=0)

plt.tight_layout()
plt.show()
# Analyzing the columns 'MiscFeature' which has 80% of missing values
print(housing_df['MiscFeature'].value_counts())
#Plotting count plot for MiscFeature column
plt.figure(figsize=(10,10))
plt.subplot(2, 2, 1)
sns.countplot(x='MiscFeature', data=housing_df)
plt.title('Distribution of MiscFeature before imputing')
plt.xticks(fontsize=11, rotation=90)
plt.yticks(fontsize=11, rotation=0)

#Instead of dropping the MiscFeature column which seems of high relevancy lets impute the missing with 
#'None' as it corresponds to nan
housing_df.loc[:,'MiscFeature'] = housing_df.loc[:,'MiscFeature'].replace(np.nan, 'None')

plt.subplot(2, 2, 2)
sns.countplot(x='MiscFeature', data=housing_df)
plt.title('Distribution of MiscFeature after imputing')
plt.xticks(fontsize=11, rotation=90)
plt.yticks(fontsize=11, rotation=0)

plt.tight_layout()
plt.show()
# Checking again the percentage of missing value
round(100*(housing_df.isnull().sum()/len(housing_df.index)), 2)
# Analyzing the columns 'LotFrontage' which has 17% of missing values
print(housing_df['LotFrontage'].value_counts())
print(housing_df['LotFrontage'].describe())
#Checking for outliers in column LotFrontage
sns.boxplot(x=housing_df['LotFrontage'])
#As there are outliers in column LotFrontage which is a continuos variable lets impute with Median values
#Plotting count plot for LotFrontage column
plt.figure(figsize=(20,15))
plt.subplot(2, 2, 1)
sns.countplot(x='LotFrontage', data=housing_df)
plt.title('Distribution of LotFrontage before imputing')
plt.xticks(fontsize=11, rotation=90)
plt.yticks(fontsize=11, rotation=0)

#Instead of dropping the LotFrontage column which seems of high relevancy lets impute the missing values with median
housing_df['LotFrontage'].replace(np.nan, housing_df['LotFrontage'].median(), inplace =True)

plt.subplot(2, 2, 2)
sns.countplot(x='LotFrontage', data=housing_df)
plt.title('Distribution of LotFrontage after imputing')
plt.xticks(fontsize=11, rotation=90)
plt.yticks(fontsize=11, rotation=0)

plt.tight_layout()
plt.show()

# Selecting columns having any missing values
miss_val_cols = list(housing_df.loc[:,(round(100*(housing_df.isnull().sum()/len(housing_df.index)), 2)>0)].columns)
miss_val_cols
# Analyzing the columns ''MasVnrType'' which has 0.5% of missing values
print(housing_df['MasVnrType'].value_counts())
#Imputing the null with mode values for the above Categorical column
housing_df['MasVnrType'].mode()[0]
housing_df['MasVnrType'].replace(np.nan,housing_df['MasVnrType'].mode()[0], inplace =True)
# Analyzing the columns 'MasVnrArea' which has 0.5% of missing values
print(housing_df['MasVnrArea'].value_counts())
#Checking for outliers in column MasVnrArea
sns.boxplot(x=housing_df['MasVnrArea'])
#Instead of dropping the LotFrontage column which seems of high relevancy lets impute the missing values with median
housing_df['MasVnrArea'].replace(np.nan, housing_df['MasVnrArea'].median(), inplace =True)
# Analyzing the columns 'BsmtQual' which has more than 2% of missing values
print(housing_df['BsmtQual'].value_counts())
#Instead of dropping the BsmtQual column which seems of high relevancy lets impute the missing values with No Basement
housing_df['BsmtQual'].replace(np.nan,'No Basement', inplace =True)
# Analyzing the columns 'BsmtCond' which has more than 2% of missing values
print(housing_df['BsmtCond'].value_counts())
#Instead of dropping the BsmtCond column which seems of high relevancy lets impute the missing values with No Basement
housing_df['BsmtCond'].replace(np.nan,'No Basement', inplace =True)
# Analyzing the columns 'BsmtExposure' which has more than 2% of missing values
print(housing_df['BsmtExposure'].value_counts())
#Instead of dropping the BsmtExposure column which seems of high relevancy lets impute the missing values with No Basement
housing_df['BsmtExposure'].replace(np.nan,'No Basement', inplace =True)
# Analyzing the columns 'BsmtFinType1' which has more than 2% of missing values
print(housing_df['BsmtFinType1'].value_counts())
#Instead of dropping the BsmtFinType1 column which seems of high relevancy lets impute the missing values with No Basement
housing_df['BsmtFinType1'].replace(np.nan,'No Basement', inplace =True)
# Analyzing the columns 'BsmtFinType2' which has more than 2% of missing values
print(housing_df['BsmtFinType2'].value_counts())
#Instead of dropping the BsmtFinType2 column which seems of high relevancy lets impute the missing values with No Basement
housing_df['BsmtFinType2'].replace(np.nan,'No Basement', inplace =True)
# Analyzing the columns 'Electrical' which has more than 0% of missing values
print(housing_df['Electrical'].value_counts())
#Imputing the null with mode values for the above Categorical column
housing_df['Electrical'].mode()[0]
housing_df['Electrical'].replace(np.nan,housing_df['Electrical'].mode()[0], inplace =True)
# Analyzing the columns 'GarageType' which has more than 5% of missing values
print(housing_df['GarageType'].value_counts())
##Instead of dropping the GarageType column which seems of high relevancy lets impute the missing values with No Garage
housing_df['GarageType'].replace(np.nan,'No Garage', inplace =True)
# Analyzing the columns 'GarageYrBlt' which has more than 5% of missing values
print(housing_df['GarageYrBlt'].value_counts())
print(housing_df['GarageYrBlt'].describe())
#Checking for outliers in column GarageYrBlt
sns.boxplot(x=housing_df['GarageYrBlt'])
#Imputing the null with mean values for the above numerical column
housing_df['GarageYrBlt'].mean()
housing_df['GarageYrBlt'].replace(np.nan, housing_df['GarageYrBlt'].mean(), inplace =True)
# Analyzing the columns 'GarageYrBlt' which has more than 5% of missing values
print(housing_df['GarageFinish'].value_counts())
##Instead of dropping the GarageFinish column which seems of high relevancy lets impute the missing values with No Garage
housing_df['GarageFinish'].replace(np.nan,'No Garage', inplace =True)
# Analyzing the columns 'GarageQual' which has more than 5% of missing values
print(housing_df['GarageQual'].value_counts())
##Instead of dropping the GarageQual column which seems of high relevancy lets impute the missing values with No Garage
housing_df['GarageQual'].replace(np.nan,'No Garage', inplace =True)
# Analyzing the columns 'GarageCond' which has more than 5% of missing values
print(housing_df['GarageCond'].value_counts())
##Instead of dropping the GarageCond column which seems of high relevancy lets impute the missing values with No Garage
housing_df['GarageCond'].replace(np.nan,'No Garage', inplace =True)
# Checking again the percentage of missing value
round(100*(housing_df.isnull().sum()/len(housing_df.index)), 2)
housing_df.describe()
housing_df.info()
Year_Month_cols = ['YrSold', 'GarageYrBlt', 'YearBuilt','YearRemodAdd', 'MoSold']
housing_df[Year_Month_cols].describe()
housing_df['#ofYrs_YrSold'] = max(housing_df['YrSold']) - housing_df['YrSold']
housing_df['#ofYrs_YrSold'].describe()
housing_df['#ofYrs_GarageYrBlt'] = max(housing_df['GarageYrBlt']) - housing_df['GarageYrBlt']
housing_df['#ofYrs_GarageYrBlt'].describe()
housing_df['#ofYrs_YearBuilt'] = max(housing_df['YearBuilt']) - housing_df['YearBuilt']
housing_df['#ofYrs_YearBuilt'].describe()
housing_df['#ofYrs_YearRemodAdd'] = max(housing_df['YearRemodAdd']) - housing_df['YearRemodAdd']
housing_df['#ofYrs_YearRemodAdd'].describe()
housing_df['#ofMths_MoSold'] = max(housing_df['MoSold']) - housing_df['MoSold']
housing_df['#ofMths_MoSold'].describe()
# Checking for Outliers
plt.figure(figsize=(12, 8), dpi=100, facecolor='w', edgecolor='k')

plt.subplot(3, 2, 1)
sns.boxplot(x=round(housing_df['#ofYrs_YrSold']))

plt.subplot(3, 2, 2)
sns.boxplot(x=housing_df['#ofYrs_GarageYrBlt'])

plt.subplot(3, 2, 3)
sns.boxplot(x=housing_df['#ofYrs_YearBuilt'])

plt.subplot(3, 2, 4)
sns.boxplot(x=housing_df['#ofYrs_YearRemodAdd'])

plt.subplot(3, 2, 5)
sns.boxplot(x=housing_df['#ofMths_MoSold'])

plt.tight_layout()

plt.show()
def Outlier_Treat_IQR(df_col):
    Q1 = 0
    Q3 = 0
    Q1 = df_col.quantile(0.05)
    Q3 = df_col.quantile(0.95)
    IQR = Q3 - Q1
    output = df_col[~((df_col < (Q1 - 1.5 * IQR)) |(df_col > (Q3 + 1.5 * IQR)))]
    return output

# Performing IQR and Assigning to the Original Column

plt.figure(figsize=(12, 8), dpi=100, facecolor='w', edgecolor='k')

#Column - '#ofYrs_GarageYrBlt'
housing_df['#ofYrs_GarageYrBlt'] = Outlier_Treat_IQR(housing_df['#ofYrs_GarageYrBlt'])
plt.subplot(2, 2, 1)
sns.boxplot(x=Outlier_Treat_IQR(housing_df['#ofYrs_GarageYrBlt']))

#Column - '#ofYrs_YearBuilt'
housing_df['#ofYrs_YearBuilt'] = Outlier_Treat_IQR(housing_df['#ofYrs_YearBuilt'])
plt.subplot(2, 2, 2)
sns.boxplot(x=Outlier_Treat_IQR(housing_df['#ofYrs_YearBuilt']))

#Column - '#ofYrs_YearRemodAdd'
housing_df['#ofYrs_YearRemodAdd'] = Outlier_Treat_IQR(housing_df['#ofYrs_YearRemodAdd'])
plt.subplot(2, 2, 3)
sns.boxplot(x=Outlier_Treat_IQR(housing_df['#ofYrs_YearRemodAdd']))

#Column - '#ofMths_MoSold'
housing_df['#ofMths_MoSold'] = Outlier_Treat_IQR(housing_df['#ofMths_MoSold'])
plt.subplot(2, 2, 4)
sns.boxplot(x=Outlier_Treat_IQR(housing_df['#ofMths_MoSold']))

plt.tight_layout()
plt.show()
print(housing_df.shape)

Year_Month_cols = ['YrSold', 'GarageYrBlt', 'YearBuilt','YearRemodAdd', 'MoSold','#ofMths_MoSold']

housing_df.drop(columns=Year_Month_cols, inplace = True)
print(housing_df.shape)
housing_df.describe()
# Checking for Outliers
plt.figure(figsize=(12, 8), dpi=100, facecolor='w', edgecolor='k')

plt.subplot(2, 2, 1)
sns.boxplot(x=(housing_df['MSSubClass']))

plt.subplot(2, 2, 2)
sns.boxplot(x=housing_df['LotFrontage'])

plt.subplot(2, 2, 3)
sns.boxplot(x=housing_df['LotArea'])

plt.subplot(2, 2, 4)
sns.boxplot(x=housing_df['OverallQual'])

plt.tight_layout()

plt.show()
# Performing IQR and Assigning to the Original Column

plt.figure(figsize=(12, 8), dpi=100, facecolor='w', edgecolor='k')

#Column - 'MSSubClass'
housing_df['MSSubClass'] = Outlier_Treat_IQR(housing_df['MSSubClass'])
plt.subplot(2, 2, 1)
sns.boxplot(x=Outlier_Treat_IQR(housing_df['MSSubClass']))

#Column - 'LotFrontage'
housing_df['LotFrontage'] = Outlier_Treat_IQR(housing_df['LotFrontage'])
plt.subplot(2, 2, 2)
sns.boxplot(x=Outlier_Treat_IQR(housing_df['LotFrontage']))

#Column - 'LotArea'
housing_df['LotArea'] = Outlier_Treat_IQR(housing_df['LotArea'])
plt.subplot(2, 2, 3)
sns.boxplot(x=Outlier_Treat_IQR(housing_df['LotArea']))\

#Column - 'OverallQual'
housing_df['OverallQual'] = Outlier_Treat_IQR(housing_df['OverallQual'])
plt.subplot(2, 2, 4)
sns.boxplot(x=Outlier_Treat_IQR(housing_df['OverallQual']))

plt.tight_layout()
plt.show()
# Checking for Outliers
plt.figure(figsize=(12, 8), dpi=100, facecolor='w', edgecolor='k')

plt.subplot(2, 2, 1)
sns.boxplot(x= housing_df['OverallCond'])

plt.subplot(2, 2, 2)
sns.boxplot(x=housing_df['MasVnrArea'])

plt.subplot(2, 2, 3)
sns.boxplot(x=housing_df['BsmtFinSF1'])

plt.subplot(2, 2, 4)
sns.boxplot(x=housing_df['BsmtFinSF2'])

plt.tight_layout()

plt.show()
# Performing IQR and Assigning to the Original Column

plt.figure(figsize=(12, 8), dpi=100, facecolor='w', edgecolor='k')

#Column - 'OverallCond'
housing_df['OverallCond'] = Outlier_Treat_IQR(housing_df['OverallCond'])
plt.subplot(2, 2, 1)
sns.boxplot(x=Outlier_Treat_IQR(housing_df['OverallCond']))

#Column - 'MasVnrArea'
housing_df['MasVnrArea'] = Outlier_Treat_IQR(housing_df['MasVnrArea'])
plt.subplot(2, 2, 2)
sns.boxplot(x=Outlier_Treat_IQR(housing_df['MasVnrArea']))

#Column - 'BsmtFinSF1'
housing_df['BsmtFinSF1'] = Outlier_Treat_IQR(housing_df['BsmtFinSF1'])
plt.subplot(2, 2, 3)
sns.boxplot(x=Outlier_Treat_IQR(housing_df['BsmtFinSF1']))\

#Column - 'BsmtFinSF2'
housing_df['BsmtFinSF2'] = Outlier_Treat_IQR(housing_df['BsmtFinSF2'])
plt.subplot(2, 2, 4)
sns.boxplot(x=Outlier_Treat_IQR(housing_df['BsmtFinSF2']))

plt.tight_layout()
plt.show()
# Checking for Outliers
plt.figure(figsize=(12, 8), dpi=100, facecolor='w', edgecolor='k')

plt.subplot(2, 2, 1)
sns.boxplot(x= housing_df['BsmtUnfSF'])

plt.subplot(2, 2, 2)
sns.boxplot(x=housing_df['TotalBsmtSF'])

plt.subplot(2, 2, 3)
sns.boxplot(x=housing_df['1stFlrSF'])

plt.subplot(2, 2, 4)
sns.boxplot(x=housing_df['2ndFlrSF'])

plt.tight_layout()

plt.show()
# Performing IQR and Assigning to the Original Column

plt.figure(figsize=(12, 8), dpi=100, facecolor='w', edgecolor='k')

#Column - 'BsmtUnfSF'
housing_df['BsmtUnfSF'] = Outlier_Treat_IQR(housing_df['BsmtUnfSF'])
plt.subplot(2, 2, 1)
sns.boxplot(x=Outlier_Treat_IQR(housing_df['BsmtUnfSF']))

#Column - 'TotalBsmtSF'
housing_df['TotalBsmtSF'] = Outlier_Treat_IQR(housing_df['TotalBsmtSF'])
plt.subplot(2, 2, 2)
sns.boxplot(x=Outlier_Treat_IQR(housing_df['TotalBsmtSF']))

#Column - '1stFlrSF'
housing_df['1stFlrSF'] = Outlier_Treat_IQR(housing_df['1stFlrSF'])
plt.subplot(2, 2, 3)
sns.boxplot(x=Outlier_Treat_IQR(housing_df['1stFlrSF']))
                                
#Column - '2ndFlrSF'
housing_df['2ndFlrSF'] = Outlier_Treat_IQR(housing_df['2ndFlrSF'])
plt.subplot(2, 2, 4)
sns.boxplot(x=Outlier_Treat_IQR(housing_df['2ndFlrSF']))

plt.tight_layout()
plt.show()
housing_df['2ndFlrSF'].describe()
# Checking for Outliers
plt.figure(figsize=(12, 8), dpi=100, facecolor='w', edgecolor='k')

plt.subplot(2, 2, 1)
sns.boxplot(x= housing_df['LowQualFinSF'])

plt.subplot(2, 2, 2)
sns.boxplot(x=housing_df['GrLivArea'])

plt.subplot(2, 2, 3)
sns.boxplot(x=housing_df['GarageArea'])

plt.subplot(2, 2, 4)
sns.boxplot(x=housing_df['WoodDeckSF'])

plt.tight_layout()

plt.show()
# Performing IQR and Assigning to the Original Column

plt.figure(figsize=(12, 8), dpi=100, facecolor='w', edgecolor='k')

#Column - 'LowQualFinSF'
housing_df['LowQualFinSF'] = Outlier_Treat_IQR(housing_df['LowQualFinSF'])
plt.subplot(2, 2, 1)
sns.boxplot(x=Outlier_Treat_IQR(housing_df['LowQualFinSF']))

#Column - 'GrLivArea'
housing_df['GrLivArea'] = Outlier_Treat_IQR(housing_df['GrLivArea'])
plt.subplot(2, 2, 2)
sns.boxplot(x=Outlier_Treat_IQR(housing_df['GrLivArea']))

#Column - 'GarageArea'
housing_df['GarageArea'] = Outlier_Treat_IQR(housing_df['GarageArea'])
plt.subplot(2, 2, 3)
sns.boxplot(x=Outlier_Treat_IQR(housing_df['GarageArea']))

#Column - 'WoodDeckSF'
housing_df['WoodDeckSF'] = Outlier_Treat_IQR(housing_df['WoodDeckSF'])
plt.subplot(2, 2, 4)
sns.boxplot(x=Outlier_Treat_IQR(housing_df['WoodDeckSF']))

plt.tight_layout()
plt.show()
# Checking for Outliers
plt.figure(figsize=(12, 8), dpi=100, facecolor='w', edgecolor='k')

plt.subplot(2, 2, 1)
sns.boxplot(x= housing_df['OpenPorchSF'])

plt.subplot(2, 2, 2)
sns.boxplot(x=housing_df['EnclosedPorch'])

plt.subplot(2, 2, 3)
sns.boxplot(x=housing_df['3SsnPorch'])

plt.subplot(2, 2, 4)
sns.boxplot(x=housing_df['ScreenPorch'])

plt.tight_layout()

plt.show()
# Performing IQR and Assigning to the Original Column

plt.figure(figsize=(12, 8), dpi=100, facecolor='w', edgecolor='k')

#Column - 'OpenPorchSF'
housing_df['OpenPorchSF'] = Outlier_Treat_IQR(housing_df['OpenPorchSF'])
plt.subplot(2, 2, 1)
sns.boxplot(x=Outlier_Treat_IQR(housing_df['OpenPorchSF']))

#Column - 'EnclosedPorch'
housing_df['EnclosedPorch'] = Outlier_Treat_IQR(housing_df['EnclosedPorch'])
plt.subplot(2, 2, 2)
sns.boxplot(x=Outlier_Treat_IQR(housing_df['EnclosedPorch']))

#Column - '3SsnPorch'
housing_df['3SsnPorch'] = Outlier_Treat_IQR(housing_df['3SsnPorch'])
plt.subplot(2, 2, 3)
sns.boxplot(x=Outlier_Treat_IQR(housing_df['3SsnPorch']))

#Column - 'ScreenPorch'
housing_df['ScreenPorch'] = Outlier_Treat_IQR(housing_df['ScreenPorch'])
plt.subplot(2, 2, 4)
sns.boxplot(x=Outlier_Treat_IQR(housing_df['ScreenPorch']))

plt.tight_layout()
plt.show()
# Checking for Outliers
plt.figure(figsize=(12, 8), dpi=100, facecolor='w', edgecolor='k')

plt.subplot(2, 2, 1)
sns.boxplot(x= housing_df['PoolArea'])

plt.subplot(2, 2, 2)
sns.boxplot(x=housing_df['MiscVal'])

plt.subplot(2, 2, 3)
sns.boxplot(x=housing_df['SalePrice'])

plt.tight_layout()

plt.show()
# Performing IQR and Assigning to the Original Column

plt.figure(figsize=(12, 8), dpi=100, facecolor='w', edgecolor='k')

#Column - 'PoolArea'
housing_df['PoolArea'] = Outlier_Treat_IQR(housing_df['PoolArea'])
plt.subplot(2, 2, 1)
sns.boxplot(x=Outlier_Treat_IQR(housing_df['PoolArea']))

#Column - 'MiscVal'
housing_df['MiscVal'] = Outlier_Treat_IQR(housing_df['MiscVal'])
plt.subplot(2, 2, 2)
sns.boxplot(x=Outlier_Treat_IQR(housing_df['MiscVal']))

#Column - 'SalePrice'
housing_df['SalePrice'] = Outlier_Treat_IQR(housing_df['SalePrice'])
plt.subplot(2, 2, 3)
sns.boxplot(x=Outlier_Treat_IQR(housing_df['SalePrice']))


plt.tight_layout()
plt.show()
# These columns have more than 85% of zero values leading to uneven distribution of data;  
Drop_Numerical_Cols = ['LowQualFinSF','PoolArea','MiscVal','EnclosedPorch','3SsnPorch','ScreenPorch','BsmtFinSF2']
round(100*((housing_df[Drop_Numerical_Cols] == 0).sum()/len(housing_df.index)), 2)
print(housing_df.shape)
housing_df.drop(columns=Drop_Numerical_Cols, inplace = True)
print(housing_df.shape)
# Re-Checking columns with any % of missing values
list(housing_df.loc[:,(round(100*(housing_df.isnull().sum()/len(housing_df.index)), 2)>0)].columns)
#Dropping null rows with null values after treating outliers
housing_df.dropna(subset=['LotFrontage'],inplace=True)
housing_df.dropna(subset=['LotArea'],inplace=True)
housing_df.dropna(subset=['MasVnrArea'],inplace=True)

housing_df.dropna(subset=['BsmtFinSF1'],inplace=True)
housing_df.dropna(subset=['TotalBsmtSF'],inplace=True)
housing_df.dropna(subset=['1stFlrSF'],inplace=True)

housing_df.dropna(subset=['GrLivArea'],inplace=True)
housing_df.dropna(subset=['WoodDeckSF'],inplace=True)
housing_df.dropna(subset=['OpenPorchSF'],inplace=True)
housing_df.dropna(subset=['SalePrice'],inplace=True)
#[housing_df[x].dropna() for x in Numeric_Cols_Null]
# Re-Checking columns with any % of missing values
list(housing_df.loc[:,(round(100*(housing_df.isnull().sum()/len(housing_df.index)), 2)>0)].columns)
sns.distplot(housing_df['SalePrice']).set(xscale = 'log')
plt.show()
from matplotlib import pyplot as plt
from scipy.stats import normaltest
import numpy as np
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='yeo-johnson')
#housing_df[['SalePrice']] = housing_df['SalePrice'].reshape(-1, 1)
pt.fit(housing_df[['SalePrice']])
transformed_data_1 = pt.transform(housing_df[['SalePrice']])
sns.distplot(transformed_data_1)
plt.show()
sns.distplot(transformed_data_1)
plt.show()
transformed_data_2 = np.log(housing_df['SalePrice'])
sns.distplot(transformed_data_2)
plt.show()
housing_df.describe()
Numerical_Var_Set1 = ['LotFrontage','LotArea', 'TotalBsmtSF', 'GrLivArea', 'GarageArea'  ,'OpenPorchSF']
Numerical_Var_Set2 = ['OverallQual','OverallCond', 'MSSubClass', 'MasVnrArea', 'BsmtFinSF1','BsmtUnfSF']
Numerical_Var_Set3= ['#ofYrs_YrSold','#ofYrs_GarageYrBlt', '#ofYrs_YearBuilt', '#ofYrs_YearRemodAdd', '1stFlrSF','2ndFlrSF']

#Plotting for the first numerical set
sns.pairplot(data = housing_df[Numerical_Var_Set1], dropna = True, kind="reg" )
plt.show()
plt.tight_layout()
#Plotting for the second numerical set
sns.pairplot(data = housing_df[Numerical_Var_Set2], dropna = True, kind="reg" )
plt.show()
plt.tight_layout()
#Plotting for the third numerical set
sns.pairplot(data = housing_df[Numerical_Var_Set3], dropna = True, kind="reg" )
plt.show()
plt.tight_layout()
plt.figure(figsize=(20, 22), dpi=100, facecolor='w', edgecolor='k')
plt.subplot(5,2,1)
plt.ylim([0, max(housing_df['SalePrice'])])
sns.boxplot(x = 'MSZoning', y = 'SalePrice', data = housing_df )
plt.xticks(fontsize=12, rotation=0)
plt.yticks(fontsize=12, rotation=0)
plt.xlabel('MSZoning',fontsize=12 )
plt.ylabel('SalePrice',fontsize=12 )
plt.title('SalePrice vs MSZoning')


plt.subplot(5,2,2)
plt.ylim([0, max(housing_df['SalePrice'])])
sns.boxplot(x = 'Street', y = 'SalePrice', data = housing_df )
plt.xticks(fontsize=12, rotation=0)
plt.yticks(fontsize=12, rotation=0)
plt.xlabel('Street',fontsize=12 )
plt.ylabel('SalePrice',fontsize=12 )
plt.title('SalePrice vs Street')

plt.subplot(5,2,3)
plt.ylim([0, max(housing_df['SalePrice'])])
sns.boxplot(x = 'LotShape', y = 'SalePrice', data = housing_df )
plt.xticks(fontsize=12, rotation=0)
plt.yticks(fontsize=12, rotation=0)
plt.xlabel('LotShape',fontsize=12 )
plt.ylabel('SalePrice',fontsize=12 )
plt.title('SalePrice vs LotShape')

plt.subplot(5,2,4)
plt.ylim([0, max(housing_df['SalePrice'])])
sns.boxplot(x = 'LandContour', y = 'SalePrice', data = housing_df )
plt.xticks(fontsize=12, rotation=0)
plt.yticks(fontsize=12, rotation=0)
plt.xlabel('LandContour',fontsize=12 )
plt.ylabel('SalePrice',fontsize=12 )
plt.title('SalePrice vs LandContour')

plt.subplot(5,2,5)
plt.ylim([0, max(housing_df['SalePrice'])])
sns.boxplot(x = 'Utilities', y = 'SalePrice', data = housing_df )
plt.xticks(fontsize=12, rotation=0)
plt.yticks(fontsize=12, rotation=0)
plt.xlabel('Utilities',fontsize=12 )
plt.ylabel('SalePrice',fontsize=12 )
plt.title('SalePrice vs Utilities')

plt.subplot(5,2,6)
plt.ylim([0, max(housing_df['SalePrice'])])
sns.boxplot(x = 'LandSlope', y = 'SalePrice', data = housing_df )
plt.xticks(fontsize=12, rotation=0)
plt.yticks(fontsize=12, rotation=0)
plt.xlabel('LandSlope',fontsize=12 )
plt.ylabel('SalePrice',fontsize=12 )
plt.title('SalePrice vs LandSlope')

plt.subplot(5,2,7)
plt.ylim([0, max(housing_df['SalePrice'])])
sns.boxplot(x = 'Neighborhood', y = 'SalePrice', data = housing_df )
plt.xticks(fontsize=12, rotation=90)
plt.yticks(fontsize=12, rotation=0)
plt.xlabel('Neighborhood',fontsize=12 )
plt.ylabel('SalePrice',fontsize=12 )
plt.title('SalePrice vs Neighborhood')

plt.subplot(5,2,8)
plt.ylim([0, max(housing_df['SalePrice'])])
sns.boxplot(x = 'Condition1', y = 'SalePrice', data = housing_df )
plt.xticks(fontsize=12, rotation=90)
plt.yticks(fontsize=12, rotation=0)
plt.xlabel('Condition1',fontsize=12 )
plt.ylabel('SalePrice',fontsize=12 )
plt.title('SalePrice vs Condition1')

plt.subplot(5,2,9)
plt.ylim([0, max(housing_df['SalePrice'])])
sns.boxplot(x = 'Condition2', y = 'SalePrice', data = housing_df )
plt.xticks(fontsize=12, rotation=90)
plt.yticks(fontsize=12, rotation=0)
plt.xlabel('Condition2',fontsize=12 )
plt.ylabel('SalePrice',fontsize=12 )
plt.title('SalePrice vs Condition2')

plt.subplot(5,2,10)
plt.ylim([0, max(housing_df['SalePrice'])])
sns.boxplot(x = 'BldgType', y = 'SalePrice', data = housing_df )
plt.xticks(fontsize=12, rotation=90)
plt.yticks(fontsize=12, rotation=0)
plt.xlabel('BldgType',fontsize=12 )
plt.ylabel('SalePrice',fontsize=12 )
plt.title('SalePrice vs BldgType')




plt.tight_layout()
plt.show()
plt.figure(figsize=(20, 22), dpi=100, facecolor='w', edgecolor='k')
plt.subplot(5,2,1)
plt.ylim([0, max(housing_df['SalePrice'])])
sns.boxplot(x = 'HouseStyle', y = 'SalePrice', data = housing_df )
plt.xticks(fontsize=12, rotation=0)
plt.yticks(fontsize=12, rotation=0)
plt.xlabel('HouseStyle',fontsize=12 )
plt.ylabel('SalePrice',fontsize=12 )
plt.title('SalePrice vs HouseStyle')


plt.subplot(5,2,2)
plt.ylim([0, max(housing_df['SalePrice'])])
sns.boxplot(x = 'RoofStyle', y = 'SalePrice', data = housing_df )
plt.xticks(fontsize=12, rotation=0)
plt.yticks(fontsize=12, rotation=0)
plt.xlabel('RoofStyle',fontsize=12 )
plt.ylabel('SalePrice',fontsize=12 )
plt.title('SalePrice vs RoofStyle')

plt.subplot(5,2,3)
plt.ylim([0, max(housing_df['SalePrice'])])
sns.boxplot(x = 'RoofMatl', y = 'SalePrice', data = housing_df )
plt.xticks(fontsize=12, rotation=0)
plt.yticks(fontsize=12, rotation=0)
plt.xlabel('RoofMatl',fontsize=12 )
plt.ylabel('SalePrice',fontsize=12 )
plt.title('SalePrice vs RoofMatl')

plt.subplot(5,2,4)
plt.ylim([0, max(housing_df['SalePrice'])])
sns.boxplot(x = 'Exterior1st', y = 'SalePrice', data = housing_df )
plt.xticks(fontsize=12, rotation=90)
plt.yticks(fontsize=12, rotation=0)
plt.xlabel('Exterior1st',fontsize=12 )
plt.ylabel('SalePrice',fontsize=12 )
plt.title('SalePrice vs Exterior1st')

plt.subplot(5,2,5)
plt.ylim([0, max(housing_df['SalePrice'])])
sns.boxplot(x = 'Exterior2nd', y = 'SalePrice', data = housing_df )
plt.xticks(fontsize=12, rotation=90)
plt.yticks(fontsize=12, rotation=0)
plt.xlabel('Exterior2nd',fontsize=12 )
plt.ylabel('SalePrice',fontsize=12 )
plt.title('SalePrice vs Exterior2nd')

plt.subplot(5,2,6)
plt.ylim([0, max(housing_df['SalePrice'])])
sns.boxplot(x = 'MasVnrType', y = 'SalePrice', data = housing_df )
plt.xticks(fontsize=12, rotation=0)
plt.yticks(fontsize=12, rotation=0)
plt.xlabel('MasVnrType',fontsize=12 )
plt.ylabel('SalePrice',fontsize=12 )
plt.title('SalePrice vs MasVnrType')

plt.subplot(5,2,7)
plt.ylim([0, max(housing_df['SalePrice'])])
sns.boxplot(x = 'ExterQual', y = 'SalePrice', data = housing_df )
plt.xticks(fontsize=12, rotation=90)
plt.yticks(fontsize=12, rotation=0)
plt.xlabel('ExterQual',fontsize=12 )
plt.ylabel('SalePrice',fontsize=12 )
plt.title('SalePrice vs ExterQual')

plt.subplot(5,2,8)
plt.ylim([0, max(housing_df['SalePrice'])])
sns.boxplot(x = 'ExterCond', y = 'SalePrice', data = housing_df )
plt.xticks(fontsize=12, rotation=90)
plt.yticks(fontsize=12, rotation=0)
plt.xlabel('ExterCond',fontsize=12 )
plt.ylabel('SalePrice',fontsize=12 )
plt.title('SalePrice vs ExterCond')

plt.subplot(5,2,9)
plt.ylim([0, max(housing_df['SalePrice'])])
sns.boxplot(x = 'Foundation', y = 'SalePrice', data = housing_df )
plt.xticks(fontsize=12, rotation=90)
plt.yticks(fontsize=12, rotation=0)
plt.xlabel('Foundation',fontsize=12 )
plt.ylabel('SalePrice',fontsize=12 )
plt.title('SalePrice vs Foundation')

plt.subplot(5,2,10)
plt.ylim([0, max(housing_df['SalePrice'])])
sns.boxplot(x = 'BsmtQual', y = 'SalePrice', data = housing_df )
plt.xticks(fontsize=12, rotation=90)
plt.yticks(fontsize=12, rotation=0)
plt.xlabel('BsmtQual',fontsize=12 )
plt.ylabel('SalePrice',fontsize=12 )
plt.title('SalePrice vs BsmtQual')




plt.tight_layout()
plt.show()
plt.figure(figsize=(20, 22), dpi=100, facecolor='w', edgecolor='k')
plt.subplot(5,2,1)
plt.ylim([0, max(housing_df['SalePrice'])])
sns.boxplot(x = 'SaleType', y = 'SalePrice', data = housing_df )
plt.xticks(fontsize=12, rotation=0)
plt.yticks(fontsize=12, rotation=0)
plt.xlabel('SaleType',fontsize=12 )
plt.ylabel('SalePrice',fontsize=12 )
plt.title('SalePrice vs SaleType')


plt.subplot(5,2,2)
plt.ylim([0, max(housing_df['SalePrice'])])
sns.boxplot(x = 'SaleCondition', y = 'SalePrice', data = housing_df )
plt.xticks(fontsize=12, rotation=0)
plt.yticks(fontsize=12, rotation=0)
plt.xlabel('SaleCondition',fontsize=12 )
plt.ylabel('SalePrice',fontsize=12 )
plt.title('SalePrice vs SaleCondition')

plt.subplot(5,2,3)
plt.ylim([0, max(housing_df['SalePrice'])])
sns.boxplot(x = 'GarageCond', y = 'SalePrice', data = housing_df )
plt.xticks(fontsize=12, rotation=0)
plt.yticks(fontsize=12, rotation=0)
plt.xlabel('GarageCond',fontsize=12 )
plt.ylabel('SalePrice',fontsize=12 )
plt.title('SalePrice vs GarageCond')

plt.subplot(5,2,4)
plt.ylim([0, max(housing_df['SalePrice'])])
sns.boxplot(x = 'GarageType', y = 'SalePrice', data = housing_df )
plt.xticks(fontsize=12, rotation=0)
plt.yticks(fontsize=12, rotation=0)
plt.xlabel('GarageType',fontsize=12 )
plt.ylabel('SalePrice',fontsize=12 )
plt.title('SalePrice vs GarageType')

plt.subplot(5,2,5)
plt.ylim([0, max(housing_df['SalePrice'])])
sns.boxplot(x = 'FireplaceQu', y = 'SalePrice', data = housing_df )
plt.xticks(fontsize=12, rotation=0)
plt.yticks(fontsize=12, rotation=0)
plt.xlabel('FireplaceQu',fontsize=12 )
plt.ylabel('SalePrice',fontsize=12 )
plt.title('SalePrice vs FireplaceQu')

plt.subplot(5,2,6)
plt.ylim([0, max(housing_df['SalePrice'])])
sns.boxplot(x = 'Functional', y = 'SalePrice', data = housing_df )
plt.xticks(fontsize=12, rotation=0)
plt.yticks(fontsize=12, rotation=0)
plt.xlabel('Functional',fontsize=12 )
plt.ylabel('SalePrice',fontsize=12 )
plt.title('SalePrice vs Functional')

plt.subplot(5,2,7)
plt.ylim([0, max(housing_df['SalePrice'])])
sns.boxplot(x = 'KitchenQual', y = 'SalePrice', data = housing_df )
plt.xticks(fontsize=12, rotation=0)
plt.yticks(fontsize=12, rotation=0)
plt.xlabel('KitchenQual',fontsize=12 )
plt.ylabel('SalePrice',fontsize=12 )
plt.title('SalePrice vs KitchenQual')

plt.subplot(5,2,8)
plt.ylim([0, max(housing_df['SalePrice'])])
sns.boxplot(x = 'Electrical', y = 'SalePrice', data = housing_df )
plt.xticks(fontsize=12, rotation=0)
plt.yticks(fontsize=12, rotation=0)
plt.xlabel('Electrical',fontsize=12 )
plt.ylabel('SalePrice',fontsize=12 )
plt.title('SalePrice vs Electrical')

plt.subplot(5,2,9)
plt.ylim([0, max(housing_df['SalePrice'])])
sns.boxplot(x = 'HeatingQC', y = 'SalePrice', data = housing_df )
plt.xticks(fontsize=12, rotation=0)
plt.yticks(fontsize=12, rotation=0)
plt.xlabel('HeatingQC',fontsize=12 )
plt.ylabel('SalePrice',fontsize=12 )
plt.title('SalePrice vs HeatingQC')

plt.subplot(5,2,10)
plt.ylim([0, max(housing_df['SalePrice'])])
sns.boxplot(x = 'CentralAir', y = 'SalePrice', data = housing_df )
plt.xticks(fontsize=12, rotation=0)
plt.yticks(fontsize=12, rotation=0)
plt.xlabel('CentralAir',fontsize=12 )
plt.ylabel('SalePrice',fontsize=12 )
plt.title('SalePrice vs CentralAir')




plt.tight_layout()
plt.show()
#Finding the correlation of variables using Heatmap
plt.figure(figsize = (18, 15))
corr=housing_df.corr()
sns.heatmap(corr, annot = True, cmap="YlGnBu")
plt.tight_layout()
#plt.show()
housing_df.columns[housing_df.nunique() == 2]
#Checking unique values in above columns
print("Street" ,housing_df['Street'].unique())
print("Utilities" ,housing_df['Utilities'].unique())
print("CentralAir" ,housing_df['CentralAir'].unique())
# Lets encode the above binary columns with 1/0
housing_df['Street'] = housing_df['Street'].map({'Pave' : 1,'Grvl': 0})
housing_df['Utilities'] = housing_df['Utilities'].map({'AllPub' : 1,'NoSeWa': 0})
housing_df['CentralAir'] = housing_df['CentralAir'].map({'Y' : 1,'N': 0})
housing_df.info()

# Indentifying dummy columns
housing_categorical_cols = list(housing_df[housing_df.columns[housing_df.nunique() > 2]].select_dtypes(exclude=['int64','float64']).columns)

housing_categorical_df = housing_df[housing_df.columns[housing_df.nunique() > 2]].select_dtypes(exclude=['int64','float64'])
housing_categorical_df.head()
#Checking unique values in above columns
print("PavedDrive" ,housing_df['PavedDrive'].unique())
print("MSZoning" ,housing_df['MSZoning'].unique())
print("Alley" ,housing_df['Alley'].unique())
print("LotShape" ,housing_df['LotShape'].unique())
print("LandContour" ,housing_df['LandContour'].unique())
print("LotConfig" ,housing_df['LotConfig'].unique())
print("LandSlope" ,housing_df['LandSlope'].unique())
print("Neighborhood" ,housing_df['Neighborhood'].unique())
print("Condition1" ,housing_df['Condition1'].unique())
print("Condition2" ,housing_df['Condition2'].unique())
print("BldgType" ,housing_df['BldgType'].unique())
print("HouseStyle" ,housing_df['HouseStyle'].unique())
print("RoofMatl" ,housing_df['RoofMatl'].unique())
print("Exterior1st" ,housing_df['Exterior1st'].unique())
print("Exterior2nd" ,housing_df['Exterior2nd'].unique())
print("MasVnrType" ,housing_df['MasVnrType'].unique())
print("ExterQual" ,housing_df['ExterQual'].unique())
print("ExterCond" ,housing_df['ExterCond'].unique())
print("Foundation" ,housing_df['Foundation'].unique())
print("BsmtQual" ,housing_df['BsmtQual'].unique())
print("BsmtCond" ,housing_df['BsmtCond'].unique())
print("BsmtExposure" ,housing_df['BsmtExposure'].unique())
print("BsmtFinType1" ,housing_df['BsmtFinType1'].unique())
print("BsmtFinType2" ,housing_df['BsmtFinType2'].unique())
print("Heating" ,housing_df['Heating'].unique())
print("HeatingQC" ,housing_df['HeatingQC'].unique())
print("Electrical" ,housing_df['Electrical'].unique())
print("KitchenQual" ,housing_df['KitchenQual'].unique())
print("Functional" ,housing_df['Functional'].unique())
print("GarageType" ,housing_df['GarageType'].unique())
print("GarageFinish" ,housing_df['GarageFinish'].unique())
print("GarageQual" ,housing_df['GarageQual'].unique())
print("GarageCond" ,housing_df['GarageCond'].unique())
print("PavedDrive" ,housing_df['PavedDrive'].unique())
print("PoolQC" ,housing_df['PoolQC'].unique())
print("Fence" ,housing_df['Fence'].unique())
print("MiscFeature" ,housing_df['MiscFeature'].unique())
print("SaleType" ,housing_df['SaleType'].unique())
print("SaleCondition" ,housing_df['SaleCondition'].unique())
# convert into dummies
housing_dummies = pd.get_dummies(housing_categorical_df, drop_first=True)
housing_dummies.head()
# drop the origininal categorical variables
print("housing_df shape before dropping the original categorical columns", housing_df.shape)
housing_df = housing_df.drop(housing_categorical_cols, axis=1)
print("housing_df shape after dropping the original categorical columns",housing_df.shape)
# concat dummy variables with X
print("housing_df shape before dummies concat and after dropping the actual categorical variables", housing_df.shape)
print("housing_dummies shape",housing_dummies.shape)
housing_df = pd.concat([housing_df, housing_dummies], axis=1)
print("housing_df shape after dummies concat", housing_df.shape)
#Checking the dataframe head
housing_df.head()
#Checking if all the variables have been handled or not
housing_df.select_dtypes(exclude=['object']).shape
# split into X and y
# Lets drop the Id and SalePrice(Which is a dependent variable) columns

#Copying the original data frame into housing_df_mod
housing_df_mod = housing_df.copy()
print("housing_df_mod shape",housing_df_mod.shape)

X = housing_df_mod.drop(columns=  ['Id','SalePrice'])
y =  np.log(housing_df['SalePrice'])
X.head()
X.info()
# split into train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    train_size=0.7,
                                                    test_size = 0.3, random_state=100)
# Lets identify the numerical columns to apply StandardScaler
(housing_df.select_dtypes(include=['int64','float64']).columns)
from sklearn.preprocessing import StandardScaler

# Apply scaler() to all the columns except the 'binary' and 'dummy' variables
scaler = StandardScaler()

num_vars = ['MSSubClass',
 'LotFrontage',
 'LotArea',
 'OverallQual',
 'OverallCond',
 'MasVnrArea',
 'BsmtFinSF1',
 'BsmtUnfSF',
 'TotalBsmtSF',
 '1stFlrSF',
 '2ndFlrSF',
 'GrLivArea',
 'BsmtFullBath',
 'BsmtHalfBath',
 'FullBath',
 'HalfBath',
 'BedroomAbvGr',
 'KitchenAbvGr',
 'TotRmsAbvGrd',
 'Fireplaces',
 'GarageCars',
 'GarageArea',
 'WoodDeckSF',
 'OpenPorchSF',
 '#ofYrs_YrSold',
 '#ofYrs_GarageYrBlt',
 '#ofYrs_YearBuilt',
 '#ofYrs_YearRemodAdd']

X_train[num_vars] = scaler.fit_transform(X_train[num_vars])
X_test[num_vars] = scaler.transform(X_test[num_vars])
X_train.shape
X_train.head()
# Calculate the VIFs for the new model
def get_vif(X_train):
    vif = pd.DataFrame()
    X = X_train
    vif['Features'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by = "VIF", ascending = False)
    return(vif)
#Checking the VIF of the entire model
#Importing the statsmodels
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

get_vif(X_train)
# list of alphas to tune
params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 
 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 
 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500, 1000 ]}


ridge = Ridge()

# cross validation
folds = 5
model_cv = GridSearchCV(estimator = ridge, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            
model_cv.fit(X_train, y_train) 
#Calculating the best param
model_cv.best_params_
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results = cv_results[cv_results['param_alpha']<=200]
cv_results.head()
# plotting mean test and train scoes with alpha 
cv_results['param_alpha'] = cv_results['param_alpha'].astype('int32')

# plotting
plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])
plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])
plt.xlabel('alpha')
plt.ylabel('Negative Mean Absolute Error')
plt.title("Negative Mean Absolute Error and alpha")
plt.legend(['train score', 'test score'], loc='upper left')
plt.show()
#As evident from the Above plot the optimal value if alpha as per Ridge Regression is 20
alpha = 20
ridge = Ridge(alpha=alpha)

ridge.fit(X_train, y_train)
ridge.coef_
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score    
ridge1 = Ridge(alpha = 20)
ridge1.fit(X_train, y_train)
print("Number of non-zero Coefficients {}".format(np.sum(ridge1.coef_!=0)))
y_pred_train = ridge1.predict(X_train)
print("RSME Train {}".format(np.sqrt(mean_squared_error(y_train, y_pred_train))))
print("R2 Score Train {}".format(r2_score(y_train, y_pred_train)))

y_pred_test = ridge1.predict(X_test)
print("RSME Test {}".format(np.sqrt(mean_squared_error(y_test, y_pred_test))))
print("R2 Score Test {}".format(r2_score(y_test, y_pred_test)))
pred = X_train.columns 
coef = pd.Series(ridge1.coef_,pred).sort_values(ascending = False)

print("**The top 10 features selected are** : ")
print(coef.head(10))
print("**The bottom 10 features selected are** : ")
print(coef.tail(10))

#Lets concatenate the top 10 and bottom 10 into a single dataset
combined_coeff = pd.concat([coef.head(10), coef.tail(10)])

#Plotting the combined_coeff
combined_coeff.plot(kind = 'barh', figsize=(10,8), title = "Model Coefficients", color='r')
lasso = Lasso()

#Defining the range of alpha parameters for Lasso Regression
params = {'alpha': [0,0.0001,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008, 0.009, 0.01]}

# Performing cross validation using GridSearchCV
model_cv = GridSearchCV(estimator = lasso, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            

#Fitting the model_cv
model_cv.fit(X_train, y_train) 
#Calculating the Best Score
model_cv.best_score_
#Calculating the best param
model_cv.best_params_
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results.head()
# plotting mean test and train scoes with alpha 
cv_results['param_alpha'] = cv_results['param_alpha'].astype('float32')

# plotting
plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])
plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])
plt.xlabel('alpha')
plt.ylabel('Negative Mean Absolute Error')

plt.title("Negative Mean Absolute Error and alpha")
plt.legend(['train score', 'test score'], loc='upper left')
plt.show()
#From the above plot the optimal value of alpha comes to be 0.001
alpha =0.001

lasso = Lasso(alpha=alpha)
        
lasso.fit(X_train, y_train) 
lasso.coef_
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score    
lasso1 = Lasso(alpha = 0.001)
lasso1.fit(X_train, y_train)
print("Number of non-zero Coefficients {}".format(np.sum(lasso1.coef_!=0)))
y_pred_train = lasso1.predict(X_train)
print("RSME Train {}".format(np.sqrt(mean_squared_error(y_train, y_pred_train))))
print("R2 Score Train {}".format(r2_score(y_train, y_pred_train)))

y_pred_test = lasso1.predict(X_test)
print("RSME Test {}".format(np.sqrt(mean_squared_error(y_test, y_pred_test))))
print("R2 Score Test {}".format(r2_score(y_test, y_pred_test)))

pred = X_train.columns 
coef = pd.Series(lasso1.coef_,pred).sort_values(ascending = False)

print("**The top 10 features selected are** : ")
print(coef.head(10))
print("**The bottom 10 features selected are** : ")
print(coef.tail(10))

#Lets concatenate the top 10 and bottom 10 into a single dataset
combined_coeff = pd.concat([coef.head(10), coef.tail(10)])

#Plotting the combined_coeff
combined_coeff.plot(kind = 'barh', figsize=(10,8), title = "Model Coefficients", color='r')
ridge2 = Ridge(alpha = 20)
ridge2.fit(X_train, y_train)
print("Number of non-zero Coefficients {}".format(np.sum(ridge2.coef_!=0)))
y_pred_train = ridge2.predict(X_train)
print("RSME Train {}".format(np.sqrt(mean_squared_error(y_train, y_pred_train))))
print("R2 Score Train {}".format(r2_score(y_train, y_pred_train)))

y_pred_test = ridge2.predict(X_test)
print("RSME Test {}".format(np.sqrt(mean_squared_error(y_test, y_pred_test))))
print("R2 Score Test {}". format(r2_score(y_test, y_pred_test)))
pred = X_train.columns 
coef = pd.Series(ridge2.coef_,pred).sort_values(ascending = False)

print("**The top 5 features selected are** : ")
print(coef.head())
print("**The bottom 5 features selected are** : ")
print(coef.tail())

#Lets concatenate the top 5 and bottom 5 into a single dataset
combined_coeff = pd.concat([coef.head(), coef.tail()])

#Plotting the combined_coeff
combined_coeff.plot(kind = 'barh', figsize=(10,8), title = "Model Coefficients", color='r')
lasso2 = Lasso(alpha = 0.002)
lasso2.fit(X_train, y_train)
print("Number of non-zero Coefficients {}".format(np.sum(lasso2.coef_!=0)))
y_pred_train = lasso2.predict(X_train)
print("RSME Train {}".format(np.sqrt(mean_squared_error(y_train, y_pred_train))))
print("R2 Score Train {}".format(r2_score(y_train, y_pred_train)))

y_pred_test = lasso2.predict(X_test)
print("RSME Test {}".format(np.sqrt(mean_squared_error(y_test, y_pred_test))))
print("R2 Score Test {}".format(r2_score(y_test, y_pred_test)))
pred = X_train.columns 
coef = pd.Series(lasso2.coef_,pred).sort_values(ascending = False)

print("**The top 5 features selected are** : ")
print(coef.head())
print("**The bottom 5 features selected are** : ")
print(coef.tail())

#Lets concatenate the top 5 and bottom 5 into a single dataset
combined_coeff = pd.concat([coef.head(), coef.tail()])

#Plotting the combined_coeff
combined_coeff.plot(kind = 'barh', figsize=(10,8), title = "Model Coefficients", color='r')
# split into X and y
# Lets drop the Id , SalePrice(Which is a dependent variable) along with columns

#Copying the original data frame into housing_df_mod
housing_df_mod = housing_df.copy()
print("housing_df_mod shape",housing_df_mod.shape)

X = housing_df_mod.drop(columns=  ['Id','SalePrice','GrLivArea','SaleType_New','Neighborhood_Crawfor', 'OverallQual','Functional_Typ'])
y =  np.log(housing_df['SalePrice'])
# split into train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    train_size=0.7,
                                                    test_size = 0.3, random_state=100)
X_train   
lasso1 = Lasso(alpha = 0.001)
lasso1.fit(X_train, y_train)
print("Number of non-zero Coefficients {}".format(np.sum(lasso1.coef_!=0)))
y_pred_train = lasso1.predict(X_train)
print("RSME Train {}".format(np.sqrt(mean_squared_error(y_train, y_pred_train))))
print("R2 Score Train {}".format(r2_score(y_train, y_pred_train)))

y_pred_test = lasso1.predict(X_test)
print("RSME Test {}".format(np.sqrt(mean_squared_error(y_test, y_pred_test))))
print("R2 Score Test {}".format(r2_score(y_test, y_pred_test)))
from sklearn.preprocessing import StandardScaler

# Apply scaler() to all the columns except the 'binary' and 'dummy' variables
scaler = StandardScaler()

num_vars = ['MSSubClass',
 'LotFrontage',
 'LotArea',
 'OverallCond',
 'MasVnrArea',
 'BsmtFinSF1',
 'BsmtUnfSF',
 'TotalBsmtSF',
 '1stFlrSF',
 '2ndFlrSF',
 'BsmtFullBath',
 'BsmtHalfBath',
 'FullBath',
 'HalfBath',
 'BedroomAbvGr',
 'KitchenAbvGr',
 'TotRmsAbvGrd',
 'Fireplaces',
 'GarageCars',
 'GarageArea',
 'WoodDeckSF',
 'OpenPorchSF',
 '#ofYrs_YrSold',
 '#ofYrs_GarageYrBlt',
 '#ofYrs_YearBuilt',
 '#ofYrs_YearRemodAdd']

X_train[num_vars] = scaler.fit_transform(X_train[num_vars])
X_test[num_vars] = scaler.transform(X_test[num_vars])
X_train.shape
lasso3 = Lasso(alpha = 0.001)
lasso3.fit(X_train, y_train)
print("Number of non-zero Coefficients {}".format(np.sum(lasso3.coef_!=0)))
y_pred_train = lasso3.predict(X_train)
print("RSME Train {}".format(np.sqrt(mean_squared_error(y_train, y_pred_train))))
print("R2 Score Train {}".format(r2_score(y_train, y_pred_train)))

y_pred_test = lasso3.predict(X_test)
print("RSME Test {}".format(np.sqrt(mean_squared_error(y_test, y_pred_test))))
print("R2 Score Test {}".format(r2_score(y_test, y_pred_test)))
pred = X_train.columns 
coef = pd.Series(lasso3.coef_,pred).sort_values(ascending = False)

print("**The top 5 features selected are** : ")
print(coef.head())
print("**The bottom 5 features selected are** : ")
print(coef.tail())

#Lets concatenate the top 5 and bottom 5 into a single dataset
combined_coeff = pd.concat([coef.head(), coef.tail()])

#Plotting the combined_coeff
combined_coeff.plot(kind = 'barh', figsize=(10,8), title = "Model Coefficients", color='r')
