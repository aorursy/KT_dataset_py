#Importing the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#Reading the file and removing the Id column

df=pd.read_csv("/kaggle/input/house1/house.csv")
df=df.drop(columns=['Id'])
df
#Getting the dimensions of the dataframe

df.shape
#Getting the various data types

df.info()
#Changing type of selected features

features = ['MSSubClass','OverallQual','OverallCond','GarageCars']

for feature in features:
    df[feature] = df[feature].astype(object)
#Seperating out numerical data

df_num=df.select_dtypes(exclude='object')
df_num
#Looking for missing values in this dataframe

df_num.isna().sum()
#Selecting out columns which have missing values.

nulldict=dict(df_num.isna().sum())
num_null=[]
for column,values in nulldict.items():
    if values!=0:
        num_null.append(column)
print("These columns have null values:",num_null)
#Plotting the distributions of these three features.

plt.figure(figsize=(16,40))

i = 1
for col in num_null:
    plt.subplot(10,4,i)
    sns.distplot(df_num[col])
    plt.xlabel(col)
    i+=1
#Plotting a heatmap of MasVnrArea against other features.

f,ax = plt.subplots(figsize=(20,2))
sns.heatmap(df_num.corr().iloc[7:8,:], annot=True, linewidths=.8, fmt= '.1f',ax=ax);
#Getting the descriptive statistical values.

df.MasVnrArea.describe()
#Replacing the missing values 

df_num.MasVnrArea.replace({np.nan:0},inplace=True)
#Plotting a heatmap of GarageYrBlt against other features.

f,ax = plt.subplots(figsize=(20,1))
sns.heatmap(df_num.corr().iloc[24:25,:], annot=True, linewidths=.8, fmt= '.1f',ax=ax);
#Getting the descriptive statistical values.

df_num.GarageYrBlt.describe()
#Replacing the missing values 

df_num.GarageYrBlt.replace({np.nan:df_num.GarageYrBlt.mean()},inplace=True)
#Plotting a heatmap of LotFrontage against other features.

f,ax = plt.subplots(figsize=(20,1))
sns.heatmap(df_num.corr().iloc[1:2,:], annot=True, linewidths=.8, fmt= '.1f',ax=ax);
#Getting the descriptive statistical values.

df_num.LotFrontage.describe()
#Replacing the missing values 

df_num.LotFrontage.replace({np.nan:df_num.LotFrontage.median()},inplace=True)
#Cheking for replacement

df_num.isnull().sum().sum()
#Seperating out object data types.

df_obj=df.select_dtypes(include="object")
df_obj
#Looking for missing values in this dataframe.

df_obj.isna().sum()
#Selecting out columns which have missing values.

nulldict=dict(df_obj.isna().sum())
num_nullobj=[]
for column,values in nulldict.items():
    if values!=0:
        num_nullobj.append(column)
print("These columns have null values:",num_nullobj)
#Dropping columns

df_obj=df_obj.drop(columns=['PoolQC','MiscFeature','Alley','Fence'])
df_obj
#Now,after the removal of four columns, our list becomes:

num_nullobj= [ 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
df_obj.Electrical.isnull().sum()
sns.countplot(df_obj['Electrical']);
df_obj.Electrical.mode()
df_obj.Electrical.replace({np.nan:'SBrkr'},inplace=True)
df_obj.MasVnrType.isnull().sum()
sns.countplot(df_obj['MasVnrType']);
df_obj.MasVnrType.replace({np.nan:'None'},inplace=True)
df_obj[['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']].isna().sum()
df[df['BsmtQual'].isnull()][['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtFinSF1',
                        'BsmtFinSF2','BsmtUnfSF','TotalBsmtSF']]
df_obj['BsmtQual'].replace({np.nan:'NA'},inplace=True)
df_obj['BsmtCond'].replace({np.nan:'NA'},inplace=True)
df_obj['BsmtExposure'].replace({np.nan:'NA'},inplace=True)
df_obj['BsmtFinType1'].replace({np.nan:'NA'},inplace=True)
df_obj['BsmtFinType2'].replace({np.nan:'NA'},inplace=True)
df_obj[['GarageType','GarageFinish','GarageQual','GarageCond']].isna().sum()
df[df['GarageType'].isnull()][['GarageType', 'GarageYrBlt', 'GarageFinish','GarageCars', 'GarageArea', 'GarageQual', 'GarageCond']]
df_obj['GarageType'].replace({np.nan:'NA'},inplace=True)
df_obj['GarageFinish'].replace({np.nan:'NA'},inplace=True)
df_obj['GarageQual'].replace({np.nan:'NA'},inplace=True)
df_obj['GarageCond'].replace({np.nan:'NA'},inplace=True)
df_obj.GarageFinish.isnull().sum()
#The various fireplace quality types and their frequency
df_obj['FireplaceQu'].value_counts()
#Comparing it with the other fire related features.
df[df['FireplaceQu'].isnull()][['Fireplaces','FireplaceQu']]
#Replacing the missing values with 'NA'
df_obj['FireplaceQu'].replace({np.nan:'NA'},inplace=True)
#Checking for replacements.
df_obj.isnull().sum()
#Creating a heat map of all the numerical features.

plt.figure(figsize=(20,20))
matrix = np.round(df_num.corr(), decimals=2)
sns.heatmap(data=matrix, linewidths=1, linecolor='black');
#Getting features that have a correlation value greater than 0.5 against sale price.

for val in range(len(matrix['SalePrice'])):
    if abs(matrix['SalePrice'].iloc[val]) > 0.5:
        print(matrix['SalePrice'].iloc[val:val+1]) 
#Relationship between sale price and the year built.
sns.scatterplot(data= df_num,x='SalePrice',y='YearBuilt');
#Relationship between sale price and the remodel date.
sns.scatterplot(data= df_num,x='SalePrice',y='YearRemodAdd');
#Relationship between sale price and total area of the basement
sns.scatterplot(data= df_num,x='SalePrice',y='TotalBsmtSF');
#Relationship between sale price and first floor area.
sns.scatterplot(data= df_num,x='SalePrice',y='1stFlrSF');
#Relationship between sale price and first floor area.
sns.scatterplot(data= df_num,x='SalePrice',y='GrLivArea');
#Relationship between sale price and full baths present.
sns.scatterplot(data= df_num,x='SalePrice',y='FullBath');
#Relationship between sale price and size of garage.
sns.scatterplot(data= df_num,x='SalePrice',y='GarageArea');
#Relationship between sale price and total rooms above grade excluding bathrooms.
sns.scatterplot(data= df_num,x='SalePrice',y='TotRmsAbvGrd');
#Analysis of sale prices across building classes.

sns.boxplot(x=df_obj['MSSubClass'],y=df_num['SalePrice']);
#Analysis of first set of features against sale price

features1=['MSZoning', 'Street', 'LotShape', 'LandContour',
       'Utilities', 'LotConfig', 'LandSlope']

plt.figure(figsize=(16,60))

i = 1
for feature in features1:
    plt.subplot(15,3,i)
    sns.boxplot(x=df_obj[feature],y=df_num['SalePrice'])
    plt.xlabel(feature)
    i+=1
#Analysis of sale prices across neighbourhoods.
plt.figure(figsize=(20,5))
ngh=sns.boxplot(x=df_obj['Neighborhood'],y=df_num['SalePrice']);
ngh.set_xticklabels(ngh.get_xticklabels(),rotation=15);
#Analysis of second set of features against sale price.

features2=['Condition1','Condition2']

plt.figure(figsize=(16,40))

i = 1
for feature in features2:
    plt.subplot(15,2,i)
    sns.boxplot(x=df_obj[feature],y=df_num['SalePrice'])
    plt.xlabel(feature)
    i+=1

#Analysis of third set of features against sale price.
features3=['BldgType', 'HouseStyle', 'OverallQual', 'OverallCond','RoofStyle', 'RoofMatl']

plt.figure(figsize=(16,60))

i = 1
for feature in features3:
    plt.subplot(15,2,i)
    sns.boxplot(x=df_obj[feature],y=df_num['SalePrice'])
    plt.xlabel(feature)
    i+=1
#Analysis of fourth set of features against sale price.
features4=['Exterior1st', 'Exterior2nd']

plt.figure(figsize=(16,60))

i = 1
for feature in features4:
    plt.subplot(15,1,i)
    sns.boxplot(x=df_obj[feature],y=df_num['SalePrice'])
    plt.xlabel(feature)
    i+=1
#Analysis of fifth set of features against sale price.
features5=['MasVnrType','ExterQual', 'ExterCond', 'Foundation']

plt.figure(figsize=(16,60))

i = 1
for feature in features5:
    plt.subplot(15,2,i)
    sns.boxplot(x=df_obj[feature],y=df_num['SalePrice'])
    plt.xlabel(feature)
    i+=1
#Analysis of sixth set of features against sale price.
features6=['BsmtQual', 'BsmtCond','BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']

plt.figure(figsize=(16,60))

i = 1
for feature in features6:
    plt.subplot(15,2,i)
    sns.boxplot(x=df_obj[feature],y=df_num['SalePrice'])
    plt.xlabel(feature)
    i+=1
#Analysis of seventh set of features against sale price.
features7=['Heating', 'HeatingQC','CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu']

plt.figure(figsize=(16,60))

i = 1
for feature in features7:
    plt.subplot(15,2,i)
    sns.boxplot(x=df_obj[feature],y=df_num['SalePrice'])
    plt.xlabel(feature)
    i+=1
#Analysis of eighth set of features against sale price.
features8=['GarageType', 'GarageFinish', 'GarageCars', 'GarageQual', 'GarageCond','PavedDrive', 'SaleType', 'SaleCondition']

plt.figure(figsize=(16,60))

i = 1
for feature in features8:
    plt.subplot(15,2,i)
    sns.boxplot(x=df_obj[feature],y=df_num['SalePrice'])
    plt.xlabel(feature)
    i+=1