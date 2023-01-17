# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
#Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

#check again the data size after dropping the 'Id' variable
print("\nThe train data size after dropping Id feature is : {} ".format(train.shape)) 
print("The test data size after dropping Id feature is : {} ".format(test.shape))
train.head(5)
train.info()
train['MSSubClass']=train['MSSubClass'].astype(int).astype(str)
train['OverallQual']=train['OverallQual'].astype(int).astype(str)
train['OverallCond']=train['OverallCond'].astype(int).astype(str)
train['BsmtFullBath']=train['BsmtFullBath'].astype(int).astype(str)
train['BsmtHalfBath']=train['BsmtHalfBath'].astype(int).astype(str)
train['FullBath']=train['FullBath'].astype(int).astype(str)
train['HalfBath']=train['HalfBath'].astype(int).astype(str)
train['BedroomAbvGr']=train['BedroomAbvGr'].astype(int).astype(str)
train['KitchenAbvGr']=train['KitchenAbvGr'].astype(int).astype(str)
train['TotRmsAbvGrd']=train['TotRmsAbvGrd'].astype(int).astype(str)
train['Fireplaces']=train['Fireplaces'].astype(int).astype(str)
train['GarageCars']=train['GarageCars'].astype(int).astype(str)
train['MoSold']=train['MoSold'].astype(int).astype(str)
train['YrSold']=train['YrSold'].astype(int).astype(str)


for cols in train.select_dtypes(include='int64'):
    plt.figure(figsize=(40,10))
    sns.jointplot(train[cols],train['SalePrice'])
    plt.show()
    
# Few Outlier Observations - Can the too big house in terms of space be excluded? or seperate models can be built?
# Below individual outlier values denote the same. Will leave it to kagglers advise. 
# And Above few variables converted should be catogorical.

# LotArea          > 100k  Too big house maybe? compared to our selling lot?
# YearBuilt        No Sign of outlier detected
# YearRemodAdd     No Sign of outlier detected
# BsmtFinSF1       > 5000    
# BsmtFinSF2       > 1400    
# TotalBsmtSF      > 6000    
# 1stFlrSF         > 4000    
# 2ndFlrSF         No Sign of outlier detected
# LowQualFinSF     > 600?
# GrLivArea        > 4000
# BsmtFullBath     = 3.0
# BsmtHalfBath     = 2.0? 
# FullBath         None
# HalfBath         None
# BedroomAbvGr     = 8?
# KitchenAbvGr     Do you think we hve any?
# TotRmsAbvGrd     14?
# Fireplaces       None
# GarageCars       None?
# GarageArea       > 1300? maybe?
# WoodDeckSF       > 700?
# OpenPorchSF      > 450
# EnclosedPorch    > 500
# 3SsnPorch        > 500
# ScreenPorch      > 400?
# Pool Area        > 600?
# MiscVal          > 8000
# MaSold 
# YrSold


#  0   MSSubClass     1460 non-null   int64  - Object
#  1   MSZoning       1460 non-null   object 
#  2   LotFrontage    1201 non-null   float64
#  3   LotArea        1460 non-null   int64  
#  4   Street         1460 non-null   object 
#  5   Alley          91 non-null     object 
#  6   LotShape       1460 non-null   object 
#  7   LandContour    1460 non-null   object 
#  8   Utilities      1460 non-null   object 
#  9   LotConfig      1460 non-null   object 
#  10  LandSlope      1460 non-null   object 
#  11  Neighborhood   1460 non-null   object 
#  12  Condition1     1460 non-null   object 
#  13  Condition2     1460 non-null   object 
#  14  BldgType       1460 non-null   object 
#  15  HouseStyle     1460 non-null   object 
#  16  OverallQual    1460 non-null   int64  - object
#  17  OverallCond    1460 non-null   int64  - object
#  18  YearBuilt      1460 non-null   int64  
#  19  YearRemodAdd   1460 non-null   int64  
#  20  RoofStyle      1460 non-null   object 
#  21  RoofMatl       1460 non-null   object 
#  22  Exterior1st    1460 non-null   object 
#  23  Exterior2nd    1460 non-null   object 
#  24  MasVnrType     1452 non-null   object 
#  25  MasVnrArea     1452 non-null   float64
#  26  ExterQual      1460 non-null   object 
#  27  ExterCond      1460 non-null   object 
#  28  Foundation     1460 non-null   object 
#  29  BsmtQual       1423 non-null   object 
#  30  BsmtCond       1423 non-null   object 
#  31  BsmtExposure   1422 non-null   object 
#  32  BsmtFinType1   1423 non-null   object 
#  33  BsmtFinSF1     1460 non-null   int64  
#  34  BsmtFinType2   1422 non-null   object 
#  35  BsmtFinSF2     1460 non-null   int64  
#  36  BsmtUnfSF      1460 non-null   int64  
#  37  TotalBsmtSF    1460 non-null   int64  
#  38  Heating        1460 non-null   object 
#  39  HeatingQC      1460 non-null   object 
#  40  CentralAir     1460 non-null   object 
#  41  Electrical     1459 non-null   object 
#  42  1stFlrSF       1460 non-null   int64  
#  43  2ndFlrSF       1460 non-null   int64  
#  44  LowQualFinSF   1460 non-null   int64  
#  45  GrLivArea      1460 non-null   int64  
#  46  BsmtFullBath   1460 non-null   int64  
#  47  BsmtHalfBath   1460 non-null   int64  
#  48  FullBath       1460 non-null   int64  
#  49  HalfBath       1460 non-null   int64  
#  50  BedroomAbvGr   1460 non-null   int64  
#  51  KitchenAbvGr   1460 non-null   int64  
#  52  KitchenQual    1460 non-null   object 
#  53  TotRmsAbvGrd   1460 non-null   int64  
#  54  Functional     1460 non-null   object 
#  55  Fireplaces     1460 non-null   int64  
#  56  FireplaceQu    770 non-null    object 
#  57  GarageType     1379 non-null   object 
#  58  GarageYrBlt    1379 non-null   float64
#  59  GarageFinish   1379 non-null   object 
#  60  GarageCars     1460 non-null   int64  
#  61  GarageArea     1460 non-null   int64  
#  62  GarageQual     1379 non-null   object 
#  63  GarageCond     1379 non-null   object 
#  64  PavedDrive     1460 non-null   object 
#  65  WoodDeckSF     1460 non-null   int64  
#  66  OpenPorchSF    1460 non-null   int64  
#  67  EnclosedPorch  1460 non-null   int64  
#  68  3SsnPorch      1460 non-null   int64  
#  69  ScreenPorch    1460 non-null   int64  
#  70  PoolArea       1460 non-null   int64  
#  71  PoolQC         7 non-null      object 
#  72  Fence          281 non-null    object 
#  73  MiscFeature    54 non-null     object 
#  74  MiscVal        1460 non-null   int64  
#  75  MoSold         1460 non-null   int64  
#  76  YrSold         1460 non-null   int64  
#  77  SaleType       1460 non-null   object 
#  78  SaleCondition  1460 non-null   object 
#  79  SalePrice      1460 non-null   int64  
# MSSubClass: Identifies the type of dwelling involved in the sale.	

#         20	1-STORY 1946 & NEWER ALL STYLES
#         30	1-STORY 1945 & OLDER
#         40	1-STORY W/FINISHED ATTIC ALL AGES
#         45	1-1/2 STORY - UNFINISHED ALL AGES
#         50	1-1/2 STORY FINISHED ALL AGES
#         60	2-STORY 1946 & NEWER
#         70	2-STORY 1945 & OLDER
#         75	2-1/2 STORY ALL AGES
#         80	SPLIT OR MULTI-LEVEL
#         85	SPLIT FOYER
#         90	DUPLEX - ALL STYLES AND AGES
#        120	1-STORY PUD (Planned Unit Development) - 1946 & NEWER
#        150	1-1/2 STORY PUD - ALL AGES
#        160	2-STORY PUD - 1946 & NEWER
#        180	PUD - MULTILEVEL - INCL SPLIT LEV/FOYER
#        190	2 FAMILY CONVERSION - ALL STYLES AND AGES
    
    
plt.figure()
train['MSSubClass'].value_counts().plot(kind='bar')

plt.figure(figsize=(20,10))
sns.swarmplot(x='MSSubClass', y = 'SalePrice', data=train)
plt.show()


# 1. This should be a categorical variable
# 2. Talks about style of the house, basic expectation of the structure, etc
# 3. Looks like 2-STORY 1946 & NEWER have houses over 700k. Could that be a outlier? will leave it to audience


# Pending Stuff

# 1. Can we perform ANVOCA? or chi square to understand the relationship between these two? We can see later. 
# MSZoning: Identifies the general zoning classification of the sale.
		
#        A	Agriculture
#        C	Commercial
#        FV	Floating Village Residential
#        I	Industrial
#        RH	Residential High Density
#        RL	Residential Low Density
#        RP	Residential Low Density Park 
#        RM	Residential Medium Density
        
        
sns.swarmplot(x ='MSZoning', y ='SalePrice', data = train) 

#  RL - "Low density residential zones" are locations intended for housing that include a lot of open space. 
#  These zones are meant for a small number of residential homes, and exclude large industries, apartment complexes, 
#  and other large structures. Home businesses, community organizations, and some types of commercial and agricultural 
#  use are allowed if they meet specific standards."  - Courtesy - https://www.gardenguides.com/
# Lets See rest of them soon!
