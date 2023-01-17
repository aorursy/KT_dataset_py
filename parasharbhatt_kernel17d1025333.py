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
# Importing necessary library

import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

import statsmodels as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.linear_model import LinearRegression

from sklearn import ensemble, tree, linear_model

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.utils import shuffle

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import f_regression

from pickle import dump , load



%matplotlib inline

import warnings

warnings.filterwarnings('ignore')



print("\n Import library is complete")
#defining parameters

inp_file="train.csv"

comp_file="test.csv"



cdir="/kaggle/input/house-prices-advanced-regression-techniques/"

os.chdir(cdir)

print (os.path)

print(os.getcwd())

#Loading training data



# Also taking copy of orig data frame to work with



train_orig = pd.read_csv(inp_file)

comp_orig = pd.read_csv(comp_file)



train=train_orig.copy() 

#df_train=train_orig.copy()    

comp=comp_orig.copy() 



# setting pandas options to display all columns and rows of dataframe



pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)
print("\n Record count in input train dataset :", train.shape[0])

print("\n Column count in input dataset :", train.shape[1])

#print("\n Column list is:\n",train.columns.values)



print("\n Record count in input comp dataset :", comp.shape[0])

print("\n Column count in input dataset :", comp.shape[1])

#print("\n Column list is:\n",comp.columns.values)





#Listing Column name and existing data type for each field



print("\n Listing Column name from train dataset with initial data type for each field : \n")

train_col_t=[]

col_t=[]

for v in train.columns.values:

    col_t.append(v)

    col_t.append(train[v].dtype)

    t=tuple(col_t)

    train_col_t.append(t)

    col_t=[]

print(train_col_t)

    

print("\n Listing Column name from comp dataset with initial data type for each field : \n")

comp_col_t=[]

col_t=[]

for v in comp.columns.values:

    col_t.append(v)

    col_t.append(comp[v].dtype)

    t=tuple(col_t)

    comp_col_t.append(t)

    col_t=[]

print(comp_col_t)

#File Wise Record Count



#print('DataType of DataFrame  train =', type(train))

print('Shape of DataFrame  train = ', train.shape)

print('Size of DataFrame train = ', train.size)

print('Dimension of DataFrame train = ', train.ndim)



print('Shape of DataFrame  comp = ', comp.shape)

print('Size of DataFrame comp = ', comp.size)

print('Dimension of DataFrame comp = ', comp.ndim)
#Generating DataFrame "NAs" for Columns with Null value count for train and comp and  dataframe



print("\n For train data: \n")

NAs = pd.concat([train.isnull().sum()], axis=1, keys=['Train'])

print(NAs[NAs.sum(axis=1) > 0])





print("\n For comp data: \n")

cNAs = pd.concat([comp.isnull().sum()], axis=1, keys=['comp'])

print(cNAs[cNAs.sum(axis=1) > 0])



#Looking at the result fields like "Alley", "Fence", "FireplaceQu" , "MiscFeature", "PoolQC" are having majority of NULL values

#Therefore they are not useful features to predict SalePrice
# Now we will go through series of data profiling and preprocessing / transformation  steps for both data sets

# For train dataset to train and test model

# For Comp dataset (original test.csv) file just to preprocess the data  / transformation and later to pass through model





"""

#ID field profiling

It represent unique sale transaction of house 

Each house is sold only once 



We can convert it as type str



"""

train['Id']=  train['Id'].astype(str)

print("\n train dataset: \n")

print( "ID field total count :",train['Id'].count() )

print("ID field Unique count :",train['Id'].nunique() )





comp['Id']=  comp['Id'].astype(str)

print("\n comp dataset: \n")

print( "ID field total count :",comp['Id'].count() )

print("ID field Unique count :",comp['Id'].nunique() )

def fn_bar(df, fld):

    plt.title("Bar graph for field : "+ fld)

    df[fld].value_counts().plot(kind='bar')

    plt.show()
"""



MSSubClass  : The building class

Nominal variable converted to Categorical variable by changing its datatype to 'str'

No missing value

"""

#If missing value comes later replace it with code :00



print("Type for train['MSSubClass'] : ", train['MSSubClass'].dtype)

print(train.groupby('MSSubClass').count()['Id'])



train['MSSubClass']=train['MSSubClass'].astype(str)

train['MSSubClass'].fillna("00", inplace=True)

print(train.groupby('MSSubClass').count()['Id'])





print("Type for comp['MSSubClass'] : ", comp['MSSubClass'].dtype)

print(comp.groupby('MSSubClass').count()['Id'])



comp['MSSubClass']=comp['MSSubClass'].astype(str)

comp['MSSubClass'].fillna("00", inplace=True)



print(comp.groupby('MSSubClass').count()['Id'])







fn_bar(train, 'MSSubClass')

"""

# MSZoning : The general zoning classification



"""

print("\n For train dataset: \n ")

print("\n Data Type for field MSZoning :",train['MSZoning'].dtype )

print("\n Train Missing value  records count for MSZoning : " , train['MSZoning'].isnull().sum())

print("\n Train MSZoning Count is :", train.groupby('MSZoning').count()['Id'])





#Replacing   null values using mode value for MSZoning



train['MSZoning'].fillna(train['MSZoning'].mode()[0], inplace=True)





print("\n For comp dataset: \n ")

print("\n Data Type for field MSZoning :", comp['MSZoning'].dtype )

print("\n comp Missing value  records count for MSZoning : " , comp['MSZoning'].isnull().sum())

print("\n comp MSZoning Count is :", comp.groupby('MSZoning').count()['Id'])

comp['MSZoning'].fillna(comp['MSZoning'].mode()[0], inplace=True)





fn_bar(train, 'MSZoning')
# LotFrontage -> Linear feet of street connected to property  -> float64



print("\n Train dataset:")

print("\n Missing value  records count for LotFrontage : ", train['LotFrontage'].isnull().sum())

print("\n Data Type for  LotFrontage ",train['LotFrontage'].dtype )

train['LotFrontage'].fillna(train['LotFrontage'].mean(), inplace=True)



print("\n comp dataset:")

print("\n Missing value  records count for LotFrontage : ", comp['LotFrontage'].isnull().sum())

print("\n Data Type for  LotFrontage ",comp['LotFrontage'].dtype )

comp['LotFrontage'].fillna(comp['LotFrontage'].mean(), inplace=True)



print("Train Missing Record count for LotFrontage after replacing missing with Mean :" , train['LotFrontage'].isnull().sum())



sns.distplot(train['LotFrontage'])

tmp1=train[['LotFrontage','LotArea','GrLivArea','SalePrice']]

pearson_corrrelation    = tmp1.corr(method="pearson");

print("\n Pearson correlation coefficient Between Above Ground Area fields and SalePrice: \n");

print(pearson_corrrelation);

sns.scatterplot(x='LotFrontage', y= 'LotArea', data=train)

#sns.pairplot(tmp1)
#  Street        Type of road access         object

# Most of the records have same street , not much useful variable for modeling...



print("\n train dataset")

print(" Data Type for field Street is: ",train['Street'].dtype )

print("\n Listing Missing value  records count :" , train['Street'].isnull().sum())

print("\n Street Count :")

print(train.loc[:, ['Id', 'Street']].groupby(['Street']).count()  )



train['Street'].fillna(train['Street'].mode(), inplace=True)





print("\n comp dataset")

print(" Data Type for field Street is: ",comp['Street'].dtype )

print("\n Listing Missing value  records count :" , comp['Street'].isnull().sum())

print("\n Street Count :")

print(comp.loc[:, ['Id', 'Street']].groupby(['Street']).count()  )

comp['Street'].fillna(comp['Street'].mode(), inplace=True)

fn_bar(train,'Street')
# Alley        Type of road access         object



print("\n for train dataset:")

print("\n Data Type for field Alley is: ",train['Alley'].dtype )

print("\n Listing Missing value  records count :", train['Alley'].isnull().sum())

train['Alley'].fillna('No Access', inplace=True)

print("\n after replacing null with No Acccess : ")

print(train.loc[:, ['Id', 'Alley']].groupby(['Alley']).count() )



      

print("\n for comp dataset:")

print("\n Data Type for field Alley is: ",comp['Alley'].dtype )

print("\n Listing Missing value  records count :", comp['Alley'].isnull().sum())

comp['Alley'].fillna('No Access', inplace=True)

print("\n after replacing null with No Acccess : ")

print(comp.loc[:, ['Id', 'Alley']].groupby(['Alley']).count() )

#  LotShape        Type of road access         object



print("\n train LotShape Count :")

print("Data Type for field LotShape is: ", train['LotShape'].dtype )

print("\n Listing Missing value  records count :",  train['LotShape'].isnull().sum())

train['LotShape'].fillna(train['LotShape'].mode()[0], inplace=True)

print(train.loc[:, ['Id','LotShape']].groupby(['LotShape']).count() )





#  LotShape        Type of road access         object



print("\n comp LotShape Count :")

print("Data Type for field LotShape is: ", comp['LotShape'].dtype )

print("\n Listing Missing value  records count :",  comp['LotShape'].isnull().sum())

comp['LotShape'].fillna(comp['LotShape'].mode()[0], inplace=True)

print(comp.loc[:, ['Id','LotShape']].groupby(['LotShape']).count() )

fn_bar(train,'LotShape')
# LandContour        Type of road access         object



print("\n Train LandContour Count :")

print("Data Type for field LandContour is: ",train['LandContour'].dtype )

print("\n Listing Missing value  records count for LandContour :" , train['LandContour'].isnull().sum())

train['LandContour'].fillna(train['LandContour'].mode()[0], inplace=True)

print(train.loc[:, ['Id','LandContour']].groupby(['LandContour']).count() )





# LandContour        Type of road access         object



print("\n comp LandContour Count :")

print("Data Type for field LandContour is: ",comp['LandContour'].dtype )

print("\n Listing Missing value  records count for LandContour :" , comp['LandContour'].isnull().sum())

comp['LandContour'].fillna(comp['LandContour'].mode()[0], inplace=True)

print(comp.loc[:, ['Id','LandContour']].groupby(['LandContour']).count() )



fn_bar(train, 'LandContour')
"""

10)        Utilities        Type of utilities available         object

11)        LotConfig        Lot configuration         object

12)        LandSlope        Slope of property         object

13)        Neighborhood        Physical locations within Ames city limits         object

14)        Condition1        Proximity to main road or railroad         object

15)        Condition2        Proximity to main road or railroad (if a second is present)         object

16)        BldgType        Type of dwelling         object

17)        HouseStyle        Style of dwelling         object





"""



for v in ['Utilities' , 'LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType', 'HouseStyle']:

    print(f"\n train dataset field {v} Count :")

    print(f"Data Type for field  {v} is: ",v, train[v].dtype )

    print("\n Listing Missing value  records count :" , train[v].isnull().count() )

    train[v].fillna(train[v].mode()[0], inplace=True)

    print(train.loc[:, ['Id', v]].groupby([v]).count() )

    

    





for v in ['Utilities' , 'LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType', 'HouseStyle']:

    print(f"\n comp dataset field {v} Count :")

    print(f"Data Type for field  {v} is: ",v, comp[v].dtype )

    print("\n Listing Missing value  records count :" , comp[v].isnull().count() )

    comp[v].fillna(comp[v].mode()[0], inplace=True)

    print(comp.loc[:, ['Id', v]].groupby([v]).count() )    

    





# Replacing Utilities missing/very less freq  values with Mode value





train['Utilities']=    train['Utilities'].apply(lambda x: 'AllPub')

print(train.loc[:, ['Id','Utilities']].groupby(['Utilities']).count() )



# Combine LandSlope  type of "Mod" and "Sev" into one say "Mod"  It will convert in only two categories for this variable



print(" \n train dataset:")

train['LandSlope']=train['LandSlope'].apply(lambda x: 'Mod' if x=='Sev' else x)

train['LandSlope'].fillna(train['LandSlope'].mode(), inplace=True)

print("\n after change in LandSlope field for Sev --> Mod")

print(train.loc[:, ['Id','LandSlope']].groupby(['LandSlope']).count() )





# Combine LandSlope  type of "Mod" and "Sev" into one say "Mod"  It will convert in only two categories for this variable



print(" \n comp dataset:")

comp['LandSlope']=comp['LandSlope'].apply(lambda x: 'Mod' if x=='Sev' else x)

comp['LandSlope'].fillna(comp['LandSlope'].mode(), inplace=True)

print("\n after change in LandSlope field for Sev --> Mod")

print(comp.loc[:, ['Id','LandSlope']].groupby(['LandSlope']).count() )

#Calculate Age of House when sold in years

#Each house is sold once (Assumption: as each house "Id" is unique in our dataset)

#instead of using more vaiables like year built and year sold, 'house age when sold' is more suitable variable





train['H_Age_When_Sold'] = train['YrSold'] - train['YearBuilt'] + 1

comp['H_Age_When_Sold'] = comp['YrSold'] - comp['YearBuilt'] + 1



sns.scatterplot(x='H_Age_When_Sold',  y='SalePrice', data=train)





comp['H_Age_When_Sold'] = comp['YrSold'] - comp['YearBuilt'] + 1

 #changed the missing values for pool quality from NA to “No Pool” to give it an appropriate level.

#these variables are not useful to build model for prediction of SalePrice as majority of values are same or missing



print('\n Train dataset: \n')

print(train[['Id','PoolArea']].groupby('PoolArea').count())

print(train[['Id','PoolQC']].groupby('PoolQC').count())



train['PoolQC'].fillna('No Pool', inplace=True)

train['PoolArea'].fillna(0, inplace=True)



print("\n  train dataset after fillna with No Pool and 0 : \n")

print(train[['Id','PoolQC']].groupby('PoolQC').count())

print(train[['Id','PoolArea']].groupby('PoolArea').count())





 #changed the missing values for pool quality from NA to “No Pool” to give it an appropriate level.

print('\n comp dataset: \n')

print(comp[['Id','PoolArea']].groupby('PoolArea').count())

print(comp[['Id','PoolQC']].groupby('PoolQC').count())



comp['PoolQC'].fillna('No Pool', inplace=True)

comp['PoolArea'].fillna(0, inplace=True)



print("\n  comp dataset after fillna with No Pool and 0 : \n")

print(comp[['Id','PoolQC']].groupby('PoolQC').count())

print(comp[['Id','PoolArea']].groupby('PoolArea').count())









#For Bsmt variables

# Missing or “NA” in'BsmtQual', 'BsmtCond', 'BsmtExposure',  'BsmtFinType1', 'BsmtFinType2' actually means “NoBSMT”

#train .loc[:,['BsmtQual', 'BsmtCond', 'BsmtExposure',  'BsmtFinType1', 'BsmtFinType2']]



for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    train[col] = train[col].fillna('NoBSMT')

    comp[col] = comp[col].fillna('NoBSMT')

    
# BsmtExposure     :  change no -> no basement



train['BsmtExposure']=train['BsmtExposure'].apply(lambda x: 'NoBSMT' if x == 'No' else x)

print("\n train dataset: \n")

print(" \n BsmtExposure After changing No  to NoBSMT  :")

print(train[['Id','BsmtExposure']].groupby('BsmtExposure').count())



# BsmtExposure     :  change no -> no basement



comp['BsmtExposure']=comp['BsmtExposure'].apply(lambda x: 'NoBSMT' if x == 'No' else x)

print("\n comp dataset: \n")

print(" \n BsmtExposure After changing No  to NoBSMT  :")

print(comp[['Id','BsmtExposure']].groupby('BsmtExposure').count())

# MasVnrType NA in all. filling with most popular values





print("\n train dataset")

train['MasVnrType'] = train['MasVnrType'].fillna(train['MasVnrType'].mode()[0])

print("\n Now we map value of field MasVnrType to 0 if 'None' otherwise 1 \n" )

train['MasVnrType'] = train['MasVnrType'].apply(lambda x: 0 if x =='None' else 1 )

print(train[['Id','MasVnrType']].groupby('MasVnrType').count())





print("\n comp dataset")

comp['MasVnrType'] = comp['MasVnrType'].fillna(comp['MasVnrType'].mode()[0])

print("\n Now we map value of field MasVnrType to 0 if 'None' otherwise 1 \n" )

comp['MasVnrType'] = comp['MasVnrType'].apply(lambda x: 0 if x =='None' else 1 )

print(comp[['Id','MasVnrType']].groupby('MasVnrType').count())



# MasVnrArea when missing or NA in all. filling with most popular values





print("\n train dataset: \n")

train['MasVnrArea'] = train['MasVnrArea'].fillna(train['MasVnrArea'].mode()[0])

#print(train[['Id','MasVnrArea']].groupby('MasVnrArea').count())

sns.scatterplot(x='MasVnrArea',  y='SalePrice', data=train)





# MasVnrArea when missing or NA in all. filling with most popular values





print("\n comp dataset: \n")

comp['MasVnrArea'] = comp['MasVnrArea'].fillna(comp['MasVnrArea'].mode()[0])

#print(comp[['Id','MasVnrArea']].groupby('MasVnrArea').count())

#sns.scatterplot(x='MasVnrArea',  y='SalePrice', data=comp)

# BsmtFinSF1 NA in all. filling with most popular values



print("\n train dataset : \n")

print("\n Missing counts for field BsmtFinSF1 :",  train['BsmtFinSF1'].isnull().sum())

print("\n Mode value for field BsmtFinSF1 : \n " , train['BsmtFinSF1'].mode()[0])

train['BsmtFinSF1'] = train['BsmtFinSF1'].fillna(train['BsmtFinSF1'].mode()[0])

#print(train[['Id','BsmtFinSF1']].groupby('BsmtFinSF1').count())







# BsmtFinSF1 NA in all. filling with most popular values



print("\n comp dataset : \n")

print("\n Missing counts for field BsmtFinSF1 :",  comp['BsmtFinSF1'].isnull().sum())

print("\n Mode value for field BsmtFinSF1 : \n " , comp['BsmtFinSF1'].mode()[0])

comp['BsmtFinSF1'] = comp['BsmtFinSF1'].fillna(comp['BsmtFinSF1'].mode()[0])

#print(comp[['Id','BsmtFinSF1']].groupby('BsmtFinSF1').count())



sns.scatterplot(x='BsmtFinSF1',  y='SalePrice', data=train)
# BsmtFinSF2 NA in all. filling with most popular values



print("\n train dataset : \n")

print("\n Missing counts for field BsmtFinSF2 :",  train['BsmtFinSF2'].isnull().sum())

print("\n Mode value for field BsmtFinSF2 : \n " , train['BsmtFinSF2'].mode()[0])

train['BsmtFinSF2'] = train['BsmtFinSF2'].fillna(train['BsmtFinSF2'].mode()[0])

print(train[['Id','BsmtFinSF2']].groupby('BsmtFinSF2').count())



# BsmtFinSF2 NA in all. filling with most popular values



print("\n comp dataset : \n")

print("\n Missing counts for field BsmtFinSF2 :",  comp['BsmtFinSF2'].isnull().sum())

print("\n Mode value for field BsmtFinSF2 : \n " , comp['BsmtFinSF2'].mode()[0])

comp['BsmtFinSF2'] = comp['BsmtFinSF2'].fillna(comp['BsmtFinSF2'].mode()[0])

print(comp[['Id','BsmtFinSF2']].groupby('BsmtFinSF2').count())





sns.scatterplot(x='BsmtFinSF2',  y='SalePrice', data=train)

# BsmtUnfSF NA in all. filling with most popular values



print("\n train dataset : \n")

print("\n Missing counts for field BsmtUnfSF :",  train['BsmtUnfSF'].isnull().sum())

print("\n Mode value for field BsmtUnfSF : \n " , train['BsmtUnfSF'].mode()[0])

train['BsmtUnfSF'] = train['BsmtUnfSF'].fillna(train['BsmtUnfSF'].mode()[0])

#print(train[['Id','BsmtUnfSF']].groupby('BsmtUnfSF').count())





# BsmtUnfSF NA in all. filling with most popular values



print("\n comp dataset : \n")

print("\n Missing counts for field BsmtUnfSF :",  comp['BsmtUnfSF'].isnull().sum())

print("\n Mode value for field BsmtUnfSF : \n " , comp['BsmtUnfSF'].mode()[0])

comp['BsmtUnfSF'] = comp['BsmtUnfSF'].fillna(comp['BsmtUnfSF'].mode()[0])

#print(comp[['Id','BsmtUnfSF']].groupby('BsmtUnfSF').count())





sns.scatterplot(x='BsmtUnfSF',  y='SalePrice', data=train)

# TotalBsmtSF NA in all. filling with most popular values



print("\n train dataset : \n")

print("\n Missing counts for field TotalBsmtSF :",  train['TotalBsmtSF'].isnull().sum())

print("\n Mode value for field TotalBsmtSF : \n " , train['TotalBsmtSF'].mode()[0])

train['TotalBsmtSF'] = train['TotalBsmtSF'].fillna(train['TotalBsmtSF'].mode()[0])

#print(train[['Id','TotalBsmtSF']].groupby('TotalBsmtSF').count())



# TotalBsmtSF NA in all. filling with most popular values



print("\n comp dataset : \n")

print("\n Missing counts for field TotalBsmtSF :",  comp['TotalBsmtSF'].isnull().sum())

print("\n Mode value for field TotalBsmtSF : \n " , comp['TotalBsmtSF'].mode()[0])

comp['TotalBsmtSF'] = comp['TotalBsmtSF'].fillna(comp['TotalBsmtSF'].mode()[0])

#print(comp[['Id','TotalBsmtSF']].groupby('TotalBsmtSF').count())





#sns.distplot(train['TotalBsmtSF'])

sns.scatterplot(x='TotalBsmtSF',  y='SalePrice', data=train)

# Generate column Has_Bsmt

print("\n train dataset : \n")

train['Has_Bsmt']=train['TotalBsmtSF'].apply(lambda x: 0 if x == 0 else 1)

print(train[['Id','Has_Bsmt']].groupby('Has_Bsmt').count())



print("\n comp dataset : \n")

comp['Has_Bsmt']=comp['TotalBsmtSF'].apply(lambda x: 0 if x == 0 else 1)

print(comp[['Id','Has_Bsmt']].groupby('Has_Bsmt').count())



#RoofMatl



#Because 98.37% of RoofMatl (roof material) is CompShg with the remaining spread across 7 categories,

#we transformed the data to "CompShg" or "Not CompShg".  Also group info for RoofStyle



print("\n train dataset: ")

print("\n Data Type of field RoofMatl : ",  train['RoofMatl'].dtype)

print("\n Record count for Missing  RoofMatl : " , train.loc[:,['RoofMatl']].isnull().sum())



train['RoofMatl'].fillna(train['RoofMatl'].mode()[0], inplace=True)

train['RoofMatl']=train['RoofMatl'].apply(lambda x: x if x =='CompShg' else 'Not CompShg')



print("\n Group count for  RoofMatl  var: ")

print(train[['Id','RoofMatl']].groupby('RoofMatl').count())



#RoofMatl



#Because 98.37% of RoofMatl (roof material) is CompShg with the remaining spread across 7 categories,

#we transformed the data to "CompShg" or "Not CompShg".  Also group info for RoofStyle



print("\n comp dataset: ")

print("\n Data Type of field RoofMatl : ",  comp['RoofMatl'].dtype)

print("\n Record count for Missing  RoofMatl : " , comp.loc[:,['RoofMatl']].isnull().sum())



comp['RoofMatl'].fillna(comp['RoofMatl'].mode()[0], inplace=True)

comp['RoofMatl']=comp['RoofMatl'].apply(lambda x: x if x =='CompShg' else 'Not CompShg')



print("\n Group count for  RoofMatl var: ")

print(comp[['Id','RoofMatl']].groupby('RoofMatl').count())



#RoofStyle



print("\n train dataset: \n")

print("\n Data Type of field RoofStyle : ",  train['RoofStyle'].dtype)

print("\n Record count for Missing  RoofStyle : " ,train.loc[:,['RoofStyle']].isnull().sum())

train['RoofStyle'].fillna(train['RoofStyle'].mode()[0], inplace=True)

train['RoofStyle']=train['RoofStyle'].apply(lambda x: x if x in ('Gable', 'Hip')  else 'Other')

print(train[['Id','RoofStyle']].groupby('RoofStyle').count())





#RoofStyle



print("\n comp dataset: \n")

print("\n Data Type of field RoofStyle : ",  comp['RoofStyle'].dtype)

print("\n Record count for Missing  RoofStyle : " ,comp.loc[:,['RoofStyle']].isnull().sum())

comp['RoofStyle'].fillna(comp['RoofStyle'].mode()[0], inplace=True)

comp['RoofStyle']=comp['RoofStyle'].apply(lambda x: x if x in ('Gable', 'Hip')  else 'Other')

print(comp[['Id','RoofStyle']].groupby('RoofStyle').count())

# Exterior1st  and  Exterior2nd

print("\n train dataset : \n")

print("\n Data Type of field Exterior1st : ",  train['Exterior1st'].dtype)

print("\n Record count for Missing  Exterior1st : " ,  train['Exterior1st'].isnull().sum())

print(train[['Id','Exterior1st']].groupby('Exterior1st').count())



print("\n Data Type of field Exterior2nd : ",  train['Exterior2nd'].dtype)

print("\n Record count for Missing  Exterior2nd : " ,  train['Exterior2nd'].isnull().sum())

print(train[['Id','Exterior2nd']].groupby('Exterior2nd').count())



train['Exterior1st'].fillna(train['Exterior1st'].mode()[0], inplace=True)

train['Exterior2nd'].fillna(train['Exterior2nd'].mode()[0], inplace=True)





# Exterior1st  and  Exterior2nd

print("\n comp dataset : \n")

print("\n Data Type of field Exterior1st : ",  comp['Exterior1st'].dtype)

print("\n Record count for Missing  Exterior1st : " ,  comp['Exterior1st'].isnull().sum())

print(comp[['Id','Exterior1st']].groupby('Exterior1st').count())



print("\n Data Type of field Exterior2nd : ",  comp['Exterior2nd'].dtype)

print("\n Record count for Missing  Exterior2nd : " ,  comp['Exterior2nd'].isnull().sum())

print(comp[['Id','Exterior2nd']].groupby('Exterior2nd').count())



comp['Exterior1st'].fillna(comp['Exterior1st'].mode()[0], inplace=True)

comp['Exterior2nd'].fillna(comp['Exterior2nd'].mode()[0], inplace=True)



#ExterQual  ,  ExterCond  , Foundation



print("\n train dataset : \n")

print("\n Data Type of field ExterQual : ",  train['ExterQual'].dtype)

print("\n Record count for Missing  ExterQual : " ,  train['ExterQual'].isnull().sum())

print(train[['Id','ExterQual']].groupby('ExterQual').count())



print("\n Data Type of field ExterCond : ",  train['ExterCond'].dtype)

print("\n Record count for Missing  ExterCond : " ,  train['ExterCond'].isnull().sum())

print(train[['Id','ExterCond']].groupby('ExterCond').count())



print("\n Data Type of field Foundation : ",  train['Foundation'].dtype)

print("\n Record count for Missing  Foundation : " ,  train['Foundation'].isnull().sum())

print(train[['Id','Foundation']].groupby('Foundation').count())



print("\n Standardizing  ExterQual and ExterCond : ")



train['ExterQual']=train['ExterQual'].apply(lambda x: x if x in ('Gd', 'TA')  else 'Other')

train['ExterCond']=train['ExterCond'].apply(lambda x: x if x in ('Gd', 'TA')  else 'Other')

train['ExterQual'].fillna(train['ExterQual'].mode()[0], inplace=True)

train['ExterCond'].fillna(train['ExterCond'].mode()[0], inplace=True)

train['Foundation'].fillna(train['Foundation'].mode()[0], inplace=True)





#ExterQual  ,  ExterCond  , Foundation

print("\n comp dataset : \n")

print("\n Data Type of field ExterQual : ",  comp['ExterQual'].dtype)

print("\n Record count for Missing  ExterQual : " ,  comp['ExterQual'].isnull().sum())

print(comp[['Id','ExterQual']].groupby('ExterQual').count())



print("\n Data Type of field ExterCond : ",  comp['ExterCond'].dtype)

print("\n Record count for Missing  ExterCond : " ,  comp['ExterCond'].isnull().sum())

print(comp[['Id','ExterCond']].groupby('ExterCond').count())



print("\n Data Type of field Foundation : ",  comp['Foundation'].dtype)

print("\n Record count for Missing  Foundation : " ,  comp['Foundation'].isnull().sum())

print(comp[['Id','Foundation']].groupby('Foundation').count())



print("\n Standardizing  ExterQual and ExterCond : ")



comp['ExterQual']=comp['ExterQual'].apply(lambda x: x if x in ('Gd', 'TA')  else 'Other')

comp['ExterCond']=comp['ExterCond'].apply(lambda x: x if x in ('Gd', 'TA')  else 'Other')

comp['ExterQual'].fillna(comp['ExterQual'].mode()[0], inplace=True)

comp['ExterCond'].fillna(comp['ExterCond'].mode()[0], inplace=True)

comp['Foundation'].fillna(comp['Foundation'].mode()[0], inplace=True)



#     Heating ,  HeatingQC , CentralAir ,Electrical





#Due to similar reasoning as RoofMatl, heating is transformed to GasA,   and other.

# Electrical is transformed in 'SBrkr' and Other

print("\n train dataset : \n")

print("\n Data Type of field Heating : ",  train['Heating'].dtype)

print("\n Record count for Missing  Heating : " ,  train['Heating'].isnull().sum())

print(train[['Id','Heating']].groupby('Heating').count())



print("\n Data Type of field HeatingQC : ",  train['HeatingQC'].dtype)

print("\n Record count for Missing  HeatingQC : " ,  train['HeatingQC'].isnull().sum())

print(train[['Id','HeatingQC']].groupby('HeatingQC').count())



print("\n Data Type of field CentralAir : ",  train['CentralAir'].dtype)

print("\n Record count for Missing  CentralAir : " ,  train['CentralAir'].isnull().sum())

print(train[['Id','CentralAir']].groupby('CentralAir').count())



print("\n Data Type of field Electrical : ",  train['Electrical'].dtype)

print("\n Record count for Missing  Electrical : " ,  train['Electrical'].isnull().sum())

print(train[['Id','Electrical']].groupby('Electrical').count())



train['Heating'].fillna(train['Heating'].mode()[0], inplace=True)

train['Electrical'].fillna(train['Electrical'].mode()[0], inplace=True)

train['Heating']=train['Heating'].apply(lambda x: x if  x in ( 'GasA' ) else 'Other' )

train['Electrical']=train['Electrical'].apply(lambda x: x if  x in ( 'SBrkr', np.NaN) else 'Other' )

train['HeatingQC'].fillna(train['HeatingQC'].mode()[0], inplace=True)

train['CentralAir'].fillna(train['CentralAir'].mode()[0], inplace=True)





print("\n comp dataset : \n")

print("\n Data Type of field Heating : ",  comp['Heating'].dtype)

print("\n Record count for Missing  Heating : " ,  comp['Heating'].isnull().sum())

print(comp[['Id','Heating']].groupby('Heating').count())



print("\n Data Type of field HeatingQC : ",  comp['HeatingQC'].dtype)

print("\n Record count for Missing  HeatingQC : " ,  comp['HeatingQC'].isnull().sum())

print(comp[['Id','HeatingQC']].groupby('HeatingQC').count())



print("\n Data Type of field CentralAir : ",  comp['CentralAir'].dtype)

print("\n Record count for Missing  CentralAir : " ,  comp['CentralAir'].isnull().sum())

print(comp[['Id','CentralAir']].groupby('CentralAir').count())



print("\n Data Type of field Electrical : ",  comp['Electrical'].dtype)

print("\n Record count for Missing  Electrical : " ,  comp['Electrical'].isnull().sum())

print(comp[['Id','Electrical']].groupby('Electrical').count())



comp['Heating'].fillna(comp['Heating'].mode()[0], inplace=True)

comp['Electrical'].fillna(comp['Electrical'].mode()[0], inplace=True)

comp['Heating']=comp['Heating'].apply(lambda x: x if  x in ( 'GasA' ) else 'Other' )

comp['Electrical']=comp['Electrical'].apply(lambda x: x if  x in ( 'SBrkr', np.NaN) else 'Other' )

comp['HeatingQC'].fillna(comp['HeatingQC'].mode()[0], inplace=True)

comp['CentralAir'].fillna(comp['CentralAir'].mode()[0], inplace=True)





"""

 #There are four variables which deal with living area: -

1stFlrSF    : First Floor square feet

2ndFlrSF    : Second floor square feet

LowQualFinSF: Low quality finished square feet (all floors)

GrLivArea   : Above grade (ground) living area square feet



#  'GrLivArea' = '1stFlrSF' + '2ndFlrSF' + 'LowQualFinSF'



Note: 

Pearson correlation coefficient r measures the strength and direction of a linear relationship 

between two variables on a scatterplot. 

The value of r is always between +1 and –1. 



Field 'LowQualFinSF' is sparsely populated and don't have much correlation with SalePrice or other variable

Field '2ndFlrSF' is above 60% with zero value (House not have second floor) and has weak positive relationship with SalePrice

                 but it also have moderate positive relationship with 'GrLivArea' and can cause multicollinearity 

                 with 'SalePrice'

Field '1stFlrSF'  is fully populated and has moderate to strong positive relationship with  'GrLivArea' and   'SalePrice'

                 and can cause multicollinearity with 'SalePrice'



Field 'GrLivArea' is sum of ('1stFlrSF' + '2ndFlrSF' + 'LowQualFinSF') and have strong positive linear relationship with 

                  SalePrice

                  

                  In general we can drop  '1stFlrSF' , '2ndFlrSF' , 'LowQualFinSF' and keep 'GrLivArea'

                  The distribution plot is slightly right skewed and can have few outliers at upper end

"""



#train[['GrLivArea' , '1stFlrSF' , '2ndFlrSF' , 'LowQualFinSF']]



#print("\n Data Type of field GrLivArea : ",  train['GrLivArea'].dtype)

#print("\n Record count for Missing  GrLivArea : " ,  train['GrLivArea'].isnull().sum())

#print("\n Record count for GrLivArea = 0 : " ,   train.loc[ train[train['GrLivArea']==0].index ,['GrLivArea']] .count()[0]  )

#print("\n Record count for GrLivArea != 0 : " ,   train.loc[ train[train['GrLivArea']!=0].index ,['GrLivArea']] .count()[0]  )



#print("\n Data Type of field 1stFlrSF : ",  train['1stFlrSF'].dtype)

#print("\n Record count for Missing  1stFlrSF : " ,  train['1stFlrSF'].isnull().sum())

#print("\n Record count for 1stFlrSF = 0 : " ,   train.loc[ train[train['1stFlrSF']==0].index ,['1stFlrSF']] .count()[0]  )



#print("\n Data Type of field 2ndFlrSF : ",  train['2ndFlrSF'].dtype)

#print("\n Record count for Missing  2ndFlrSF : " ,  train['2ndFlrSF'].isnull().sum())

#print("\n Record count for 2ndFlrSF = 0 : " ,   train.loc[ train[train['2ndFlrSF']==0].index ,['2ndFlrSF']] .count()[0]  )

#print("\n Record count for 2ndFlrSF != 0 : " ,   train.loc[ train[train['2ndFlrSF']!=0].index ,['2ndFlrSF']] .count()[0]  )





#print("\n Data Type of field LowQualFinSF : ",  train['LowQualFinSF'].dtype)

#print("\n Record count for Missing  LowQualFinSF : " ,  train['LowQualFinSF'].isnull().sum())

#print("\n Record count for LowQualFinSF = 0 : " ,   train.loc[ train[train['LowQualFinSF']==0].index ,['LowQualFinSF']] .count()[0]  )

#print("\n Record count for LowQualFinSF != 0 : " ,   train.loc[ train[train['LowQualFinSF']!=0].index ,['LowQualFinSF']] .count()[0]  )



#sns.scatterplot(x='1stFlrSF',  y='GrLivArea', data=train)

#  'GrLivArea' = '1stFlrSF' + '2ndFlrSF' + 'LowQualFinSF'

print("\n train dataset: \n")

train['1stFlrSF'].fillna(train['1stFlrSF'].mode()[0], inplace=True)

train['2ndFlrSF'].fillna(train['2ndFlrSF'].mode()[0], inplace=True)

train['LowQualFinSF'].fillna(train['LowQualFinSF'].mode()[0], inplace=True)

train['GrLivArea'].fillna(train['GrLivArea'].mode()[0], inplace=True)







tmp1=train[['GrLivArea','1stFlrSF','2ndFlrSF', 'LowQualFinSF','SalePrice']]

pearson_corrrelation    = tmp1.corr(method="pearson");

print("\n Pearson correlation coefficient Between Above Ground Area fields and SalePrice: \n");

print(pearson_corrrelation);



print("\n comp dataset: \n")

comp['1stFlrSF'].fillna(comp['1stFlrSF'].mode()[0], inplace=True)

comp['2ndFlrSF'].fillna(comp['2ndFlrSF'].mode()[0], inplace=True)

comp['LowQualFinSF'].fillna(comp['LowQualFinSF'].mode()[0], inplace=True)

comp['GrLivArea'].fillna(comp['GrLivArea'].mode()[0], inplace=True)



print("\n train dataset: \n")

sns.pairplot( data=tmp1)

#The distribution plot for 'GrLivArea' is slightly right skewed and can have few outliers at upper end



# As advised in the documentation for the original project, we have dropped all houses with an above ground living area 

#greater than 4000 square feet, dropping a total of 5 observatinons. 

#Three of them are true outliers (Partial Sales that likely don’t represent actual market values) 

# and two of them are simply unusual sales (very large houses priced relatively appropriately).



p01=train['GrLivArea'].quantile(0.01)

print("p01 = ",p01)

p25=train['GrLivArea'].quantile(0.25)

print("p25 = ",p25)

p75=train['GrLivArea'].quantile(0.75)

print("p75 = ",p75)

IQR= np.absolute(p75-p25)



print("IQR = ",IQR)

GrLivArea_lb= p25 - (1.5 * IQR)

print("GrLivArea_lb = ",GrLivArea_lb)

GrLivArea_ub= p75 + (1.5 * IQR)



print("GrLivArea_ub = ",GrLivArea_ub)



print(train['GrLivArea'].describe())



#print("\n Number of records which are outliers for field GrLivArea and can be deleted : \n  ")

#print( train.loc[  train[  (train['GrLivArea'] > GrLivArea_ub) | (train['GrLivArea'] < GrLivArea_lb) ].index, :].count()['Id'])

#print( train.loc[  train[  (train['GrLivArea'] > GrLivArea_ub) | (train['GrLivArea'] < GrLivArea_lb) ].index,  [ 'Id','GrLivArea','1stFlrSF','2ndFlrSF', 'LowQualFinSF','SalePrice']] )



#print("\n OR Number of records which are 1stFlrSF > 4000 and can be deleted : \n  ")

#print( train.loc[  train[  (train['1stFlrSF'] > 4000) ].index, :].count()['Id'])

#print( train.loc[  train[  (train['1stFlrSF'] > 4000) ].index,  [ 'Id','GrLivArea','1stFlrSF','2ndFlrSF', 'LowQualFinSF','SalePrice']] )



#sns.distplot(train['GrLivArea'])



#Incase if there is missing value we replace it with 0



train['1stFlrSF'].fillna(0,inplace=True)

train['2ndFlrSF'].fillna(0,inplace=True)

train['LowQualFinSF'].fillna(0,inplace=True)

train['GrLivArea'].fillna(0,inplace=True)



comp['1stFlrSF'].fillna(0,inplace=True)

comp['2ndFlrSF'].fillna(0,inplace=True)

comp['LowQualFinSF'].fillna(0,inplace=True)

comp['GrLivArea'].fillna(0,inplace=True)



"""

Bathroom variables:



BsmtFullBath  : Basement full bathrooms

BsmtHalfBath  : Basement half bathrooms

FullBath  : Full bathrooms above grade

HalfBath  : Half baths above grade



For simplicity let's generate variable and compare correlation of it with all bathroom variables and SalePrice



'TotalBath' = 'FullBath' + 'BsmtFullBath' + (1/2)*( 'HalfBath'  + 'BsmtHalfBath' )



"""



#print("\n Data Type of field BsmtFullBath : ",  train['BsmtFullBath'].dtype)

#print("\n Record count for Missing  BsmtFullBath : " ,  train['BsmtFullBath'].isnull().sum())

#print("\n Stat for variable BsmtFullBath : \n",  train['BsmtFullBath'].describe())

#print("\n Record count for BsmtFullBath = 0 : " ,   train.loc[ train[train['BsmtFullBath']==0].index ,['BsmtFullBath']] .count()[0]  )

#print("\n Record count for BsmtFullBath != 0 : " ,   train.loc[ train[train['BsmtFullBath']!=0].index ,['BsmtFullBath']] .count()[0]  )



#print("\n Data Type of field BsmtHalfBath : ",  train['BsmtHalfBath'].dtype)

#print("\n Record count for Missing  BsmtHalfBath : " ,  train['BsmtHalfBath'].isnull().sum())

#print("\n Stat for variable BsmtHalfBath : \n",  train['BsmtHalfBath'].describe())

#print("\n Record count for BsmtHalfBath = 0 : " ,   train.loc[ train[train['BsmtHalfBath']==0].index ,['BsmtHalfBath']] .count()[0]  )

#print("\n Record count for BsmtHalfBath != 0 : " ,   train.loc[ train[train['BsmtHalfBath']!=0].index ,['BsmtHalfBath']] .count()[0]  )



#print("\n Counts for Basement having Full or half bath :", train.loc[  train[  (train['BsmtFullBath'] != 0) | (train['BsmtHalfBath'] != 0) ].index,  [ 'Id','BsmtFullBath','BsmtHalfBath']].count()['Id'] )

#print( train.loc[  train[  (train['BsmtFullBath'] != 0) | (train['BsmtHalfBath'] != 0) ].index,  [ 'Id','BsmtFullBath','BsmtHalfBath']] )



#print("\n Data Type of field FullBath : ",  train['FullBath'].dtype)

#print("\n Record count for Missing  FullBath : " ,  train['FullBath'].isnull().sum())

#print("\n Stat for variable FullBath : \n",  train['FullBath'].describe())

#print("\n Record count for FullBath = 0 : " ,   train.loc[ train[train['FullBath']==0].index ,['FullBath']] .count()[0]  )

#print("\n Record count for FullBath != 0 : " ,   train.loc[ train[train['FullBath']!=0].index ,['FullBath']] .count()[0]  )



#print("\n Data Type of field HalfBath : ",  train['HalfBath'].dtype)

#print("\n Record count for Missing  HalfBath : " ,  train['HalfBath'].isnull().sum())

#print("\n Stat for variable HalfBath : \n",  train['HalfBath'].describe())

#print("\n Record count for HalfBath = 0 : " ,   train.loc[ train[train['HalfBath']==0].index ,['HalfBath']] .count()[0]  )

#print("\n Record count for HalfBath != 0 : " ,   train.loc[ train[train['HalfBath']!=0].index ,['HalfBath']] .count()[0]  )





#print("\n Counts for Upper Ground Area having Full or half bath :", train.loc[  train[  (train['FullBath'] != 0) | (train['HalfBath'] != 0) ].index,  [ 'Id','FullBath','HalfBath']].count()['Id'] )



train['TotalBath'] = train.apply( lambda row: row.FullBath  + row.BsmtFullBath + ( (row.BsmtHalfBath/2) + (row.HalfBath/2) )  , axis=1 )



#print("\n Records of upper ground area having Fullbath = 0 : \n")

#print( train.loc[  train[  (train['FullBath'] == 0) ].index,  [ 'Id','TotalBath','BsmtFullBath','BsmtHalfBath', 'FullBath' , 'HalfBath', 'HouseStyle']] )



#print("\n Data Type of field TotalBath : ",  train['TotalBath'].dtype)

#print("\n Record count for Missing  TotalBath : " ,  train['TotalBath'].isnull().sum())

#print("\n Stat for variable TotalBath : \n",  train['TotalBath'].describe())

#print("\n Record count for TotalBath = 0 : " ,   train.loc[ train[train['TotalBath']==0].index ,['TotalBath']] .count()[0]  )

#print("\n Record count for TotalBath != 0 : " ,   train.loc[ train[train['TotalBath']!=0].index ,['TotalBath']] .count()[0]  )



#Incase if there is missing value we replace it with 0



train['BsmtFullBath'].fillna(0, inplace=True)

train['BsmtHalfBath'].fillna(0, inplace=True)

train['FullBath'].fillna(0, inplace=True)

train['HalfBath'].fillna(0, inplace=True)

train['TotalBath'].fillna(0, inplace=True)



comp['TotalBath'] = comp.apply( lambda row: row.FullBath  + row.BsmtFullBath + ( (row.BsmtHalfBath/2) + (row.HalfBath/2) )  , axis=1 )

comp['BsmtFullBath'].fillna(0, inplace=True)

comp['BsmtHalfBath'].fillna(0, inplace=True)

comp['FullBath'].fillna(0, inplace=True)

comp['HalfBath'].fillna(0, inplace=True)

comp['TotalBath'].fillna(0, inplace=True)





#sns.distplot(train['TotalBath'])



fn_bar(train , 'TotalBath')
"""

 We can see that descrete vars TotalBath and FullBath having moderate to good positive

 correlation with SalePrice so we can keep one of them to avoid multicollinearity and we will keep

 TotalBath

 

 The other bathroom variables have low positive/negative correlation with SalePrice

 

 So we will keep at TotalBath variable as representative of all Bathroom variables and

 drop variables 'BsmtFullBath' , 'BsmtHalfBath' ,  'FullBath' , 'HalfBath'

    

"""







tmp1= train[ [ 'BsmtFullBath','BsmtHalfBath', 'FullBath' , 'HalfBath', 'TotalBath','SalePrice']] 

pearson_corrrelation    = tmp1.corr(method="pearson");

print("\n Pearson correlation coefficient Between Bathroom variables and SalePrice: \n");

print(pearson_corrrelation);



#sns.pairplot(tmp1)

"""

BedroomAbvGr : Number of bedrooms above basement level



We found that variable 'BedroomAbvGr' has medium positive correlation with 'GrLivArea'

    but weak positive relation with SalePrice



Variable 'BedroomAbvGr' can be dropped



"""



#print("\n Data Type of field BedroomAbvGr : ",  train['BedroomAbvGr'].dtype)

#print("\n Record count for Missing  BedroomAbvGr : " ,  train['BedroomAbvGr'].isnull().sum())

#print("\n Record count for BedroomAbvGr = 0 : " ,   train.loc[ train[train['BedroomAbvGr']==0].index ,['BedroomAbvGr']] .count()[0]  )

#print("\n Record count for BedroomAbvGr != 0 : " ,   train.loc[ train[train['BedroomAbvGr']!=0].index ,['BedroomAbvGr']] .count()[0]  )



#print("Grouping on number of bedrooms above ground:\n")

#print( train[['BedroomAbvGr','Id']].groupby('BedroomAbvGr').count())



#print("\n Stat for variable BedroomAbvGr : \n",  train['BedroomAbvGr'].describe())





#Incase if there is missing value we replace it with 0



train['BedroomAbvGr'].fillna(0,inplace=True)

comp['BedroomAbvGr'].fillna(0,inplace=True)





tmp1= train[ [ 'GrLivArea' , 'BedroomAbvGr', 'TotalBath','SalePrice']] 

pearson_corrrelation    = tmp1.corr(method="pearson");

print("\n Pearson correlation coefficient Between BedroomAbvGr,TotalBath variables and SalePrice: \n");

print(pearson_corrrelation);



sns.pairplot(tmp1)


"""

Kitchen variables: There are two kitchen variables



KitchenAbvGr  :  Number of kitchens

KitchenQual  :  Kitchen quality





Variable KitchenAbvGr has very low negative correlation with SalePrice

Majority of House have 1 Kitchen. Hence this variable is not a good predictor for SalePrice and can be dropped.





"""

#print("\n Data Type of field KitchenQual : ",  train['KitchenQual'].dtype)

#print("\n Record count for Missing  KitchenQual : " ,  train['KitchenQual'].isnull().sum())

#print("Grouping on Kitchen Quality for kitchen above ground:\n")

#print( train[['KitchenQual','KitchenAbvGr','Id']].groupby(['KitchenQual','KitchenAbvGr']).count()['Id'])



#print("\n Data Type of field KitchenAbvGr : ",  train['KitchenAbvGr'].dtype)

#print("\n Record count for Missing  KitchenAbvGr : " ,  train['KitchenAbvGr'].isnull().sum())

#print("\n Record count for KitchenAbvGr = 0 : " ,   train.loc[ train[train['KitchenAbvGr']==0].index ,['KitchenAbvGr']] .count()[0]  )

#print("\n Record count for KitchenAbvGr != 0 : " ,   train.loc[ train[train['KitchenAbvGr']!=0].index ,['KitchenAbvGr']] .count()[0]  )



#print("Grouping on number of Kitchen above ground:\n")

#print( train[['KitchenAbvGr','Id']].groupby('KitchenAbvGr').count())





#Incase if there is missing value we replace it with 0



train['KitchenAbvGr'].fillna(0,inplace=True)

train['KitchenQual'].fillna(train['KitchenQual'].mode()[0], inplace=True)

comp['KitchenAbvGr'].fillna(0,inplace=True)

comp['KitchenQual'].fillna(train['KitchenQual'].mode()[0], inplace=True)





#tmp1= train[ ['KitchenAbvGr', 'GrLivArea' , 'BedroomAbvGr', 'TotalBath','SalePrice']] 

#pearson_corrrelation    = tmp1.corr(method="pearson");

#print("\n Pearson correlation coefficient Between KitchenAbvGr, GrLivArea, BedroomAbvGr,TotalBath variables and SalePrice: \n");

#print(pearson_corrrelation);



#sns.pairplot(tmp1)

"""

TotRmsAbvGrd : Total rooms above grade (does not include bathrooms)



Variable 'TotRmsAbvGrd' has moderate positive corelation with 'BedroomAbvGr', 'SalePrice' and strong positive

          coorelation with 'GrLivArea'

          It can be dropped to avoid multicollinearity

"""

#print("\n Data Type of field TotRmsAbvGrd : ",  train['TotRmsAbvGrd'].dtype)

#print("\n Record count for Missing  TotRmsAbvGrd : " ,  train['TotRmsAbvGrd'].isnull().sum())

#print("\n Record count for TotRmsAbvGrd = 0 : " ,   train.loc[ train[train['TotRmsAbvGrd']==0].index ,['TotRmsAbvGrd']].count()[0]  )

#print("\n Record count for TotRmsAbvGrd != 0 : " ,   train.loc[ train[train['TotRmsAbvGrd']!=0].index ,['TotRmsAbvGrd']].count()[0]  )



#print("Grouping on number of Kitchen above ground:\n")

#print( train[['TotRmsAbvGrd','Id']].groupby('TotRmsAbvGrd').count())





#Incase if there is missing value we replace it with 0



train['TotRmsAbvGrd'].fillna(0,inplace=True)

comp['TotRmsAbvGrd'].fillna(0,inplace=True)





#tmp1= train[ ['TotRmsAbvGrd','KitchenAbvGr', 'BedroomAbvGr','GrLivArea' , 'TotalBath','SalePrice']] 

#pearson_corrrelation    = tmp1.corr(method="pearson");

#print("\n Pearson correlation coefficient Between TotRmsAbvGrd ,KitchenAbvGr, GrLivArea, BedroomAbvGr,TotalBath variables and SalePrice: \n");

#print(pearson_corrrelation);



#sns.pairplot(tmp1)

"""

Functional : Home functionality rating



Majority of values  (Over 90%) are of the type = 'Typ' so converting the variable into Binary with values

as 'Typ= 1' and NonTyp=0



IT is not a good predictor variable looking at Stats of Min and Max of other NonTyp functional ratings



"""



#print("\n Data Type of field Functional : ",  train['Functional'].dtype)

#print("\n Record count for Missing  Functional : " ,  train['Functional'].isnull().sum())

#print("Grouping on Functional rating of House:\n")

#print( train[['Functional','Id']].groupby(['Functional']).count()['Id'])





#Incase if there is missing value we replace it with 0



train['Functional'].fillna(train['Functional'].mode()[0], inplace=True)

s=train['Functional'].mode()[0]

#print ('Mode = ', s)

train['Functional'] = train['Functional'].apply(lambda x: x if x == s else ('Non'+s) )

print(" \n train dataset:")

print (train[['Functional','Id']].groupby(['Functional']).count()['Id'])





comp['Functional'].fillna(comp['Functional'].mode()[0], inplace=True)

sc=comp['Functional'].mode()[0]

#print ('Mode = ', sc)

comp['Functional'] = comp['Functional'].apply(lambda x: x if x == sc else ('Non'+sc) )

print(" \n comp dataset:")

print (comp[['Functional','Id']].groupby(['Functional']).count()['Id'])



#print("\n Max sale price by Function : \n")

#print (train[['Functional','SalePrice']].groupby(['Functional']).max()['SalePrice'])

#print("\n Min sale price by Function : \n")

#print (train[['Functional','SalePrice']].groupby(['Functional']).min()['SalePrice'])

#print(train.loc[ train[   train[ 'Functional' ]!=s].index, ['Functional','SalePrice','HouseStyle']].sort_values(by='SalePrice'))



#train[ 'Functional'].value_counts().plot(kind='bar')

      
"""



FireplaceQu  : Fireplace quality



Fireplaces   : Number of fireplaces





Looking at the counts for variable "FireplaceQu" and "Fireplaces", approx 45% of house don't have FirePlace 

and those who have, majority are having only one Fireplace



So missing values for "FireplaceQu" can be filled with 'NoFirePlace'

Fireplaces variable is discrete and has weak to moderate level of positive correlation with SalePrice



These variables can be dropped and binary variable can be added as Has_Fireplace

"""





#print("\n Data Type of field FireplaceQu : ",  train['FireplaceQu'].dtype)

#print("\n Record count for Missing  FireplaceQu : " ,  train['FireplaceQu'].isnull().sum())





#print("\n Data Type of field Fireplaces : ",  train['Fireplaces'].dtype)

#print("\n Record count for Missing  Fireplaces : " ,  train['Fireplaces'].isnull().sum())

#print("\n Record count for Fireplaces = 0 : " ,   train.loc[ train[train['Fireplaces']==0].index ,['Fireplaces']] .count()[0]  )

#print("\n Record count for Fireplaces != 0 : " ,   train.loc[ train[train['Fireplaces']!=0].index ,['Fireplaces']] .count()[0]  )

print("\n train dataset \n")

print("Grouping on Fireplace Quality :\n")

print( train[['FireplaceQu','Id']].groupby(['FireplaceQu']).count()['Id'])

print("Grouping on number of Fireplace:\n")

print( train[['Fireplaces','Id']].groupby('Fireplaces').count())

train['Fireplaces'].fillna(0,inplace=True)

train['FireplaceQu'].fillna('NoFirePlace', inplace=True)



print("\n comp dataset \n")

print("Grouping on Fireplace Quality :\n")

print( comp[['FireplaceQu','Id']].groupby(['FireplaceQu']).count()['Id'])

print("Grouping on number of Fireplace:\n")

print( comp[['Fireplaces','Id']].groupby('Fireplaces').count())

comp['Fireplaces'].fillna(0,inplace=True)

comp['FireplaceQu'].fillna('NoFirePlace', inplace=True)





#print("Grouping on Fireplace Quality :\n")

#print( train[['FireplaceQu','Id']].groupby(['FireplaceQu']).count()['Id'])

#print("Grouping on Fireplaces  :\n")

#print( train[['Fireplaces','Id']].groupby(['Fireplaces']).count()['Id'])



#tmp1= train[ ['Fireplaces', 'GrLivArea' ,  'TotalBath','SalePrice']] 

#pearson_corrrelation    = tmp1.corr(method="pearson");

#print("\n Pearson correlation coefficient Between Fireplaces, GrLivArea, TotalBath variables and SalePrice: \n");

#print(pearson_corrrelation);

#print("\n Record  for Fireplaces == 3 : " ,     )

#print(train.loc[ train[train['Fireplaces'] == 3].index ,['Fireplaces','HouseStyle','SalePrice']] )

#sns.pairplot(tmp1)



print("\n Train: \n")

train['Has_FirePlace']= train['Fireplaces'].apply(lambda x : 1 if x!=0 else 0)

print( train[['Has_FirePlace','Id']].groupby(['Has_FirePlace']).count()['Id'])



print("\n comp: \n")

comp['Has_FirePlace']= comp['Fireplaces'].apply(lambda x : 1 if x!=0 else 0)

print( comp[['Has_FirePlace','Id']].groupby(['Has_FirePlace']).count()['Id'])



"""

There are 7 Garage Variables :-->

GarageType     : Garage location   , nominal variable , six garage levels - Missing/NA for no garage   , Missing/ NA (81)

GarageQual     : Garage quality , ordinal   , five levels from poor to excellent - NA for, Missing/ NA (81)

GarageCond     : Garage condition ,five levels from poor to excellent - NA for no garage , Missing/ NA (81)

GarageFinish   : Interior finish of the garage , three levels: unfinished, rough finished, finished , Missing/ NA (81)



GarageYrBlt    : Year garage was built , discrete

GarageCars     : Size of garage in car capacity

GarageArea     : Size of garage in square feet



'GarageType' and 'GarageFinish' looks influential , whereas 

'GarageQual'  & 'GarageCond' are morethan 90% with values 'TA' and not good effective/ predictor var

'GarageYrBlt' normally not that significant compare to Type and Finishing and can be dropped



'GarageCars' and  'GarageArea' both have strong positive correlation with each other and moderate to 

            strong coorelation with SalePrice. so we can keep only one of it.

I decided to keep GarageArea over GarageCars due to its contineous nature.



"""

#print("\n Data Type of field GarageType : ",  train['GarageType'].dtype)

#print("\n Record count for Missing  GarageType : " ,  train['GarageType'].isnull().sum())

#print("\n Missing value is replaced with 'No Garage' : ")



train['GarageType'].fillna('No Garage',inplace=True)

comp['GarageType'].fillna('No Garage',inplace=True)



#print("Grouping on GarageType Quality :\n")

#print( train[['GarageType','Id']].groupby(['GarageType']).count()['Id'])





#print("\n Data Type of field GarageQual : ",  train['GarageQual'].dtype)

#print("\n Record count for Missing  GarageQual : " ,  train['GarageQual'].isnull().sum())

#print("\n Missing value is replaced with 'No Garage' : ")



train['GarageQual'].fillna('No Garage',inplace=True)

comp['GarageQual'].fillna('No Garage',inplace=True)



#print("Grouping on GarageQual Quality :\n")

#print( train[['GarageQual','Id']].groupby(['GarageQual']).count()['Id'])



#print("\n Data Type of field GarageCond : ",  train['GarageCond'].dtype)

#print("\n Record count for Missing  GarageCond : " ,  train['GarageCond'].isnull().sum())

#print("\n Missing value is replaced with 'No Garage' : ")



train['GarageCond'].fillna('No Garage',inplace=True)

comp['GarageCond'].fillna('No Garage',inplace=True)



#print("Grouping on GarageCond Quality :\n")

#print( train[['GarageCond','Id']].groupby(['GarageCond']).count()['Id'])





#print("\n Data Type of field GarageFinish : ",  train['GarageFinish'].dtype)

#print("\n Record count for Missing  GarageFinish : " ,  train['GarageFinish'].isnull().sum())

#print("\n Missing value is replaced with 'No Garage' : ")



train['GarageFinish'].fillna('No Garage',inplace=True)

comp['GarageFinish'].fillna('No Garage',inplace=True)



#print("Grouping on GarageFinish Quality :\n")

#print( train[['GarageFinish','Id']].groupby(['GarageFinish']).count()['Id'])



#print("\n Data Type of field GarageCars : ",  train['GarageCars'].dtype)

#print("\n Record count for Missing  GarageCars : " ,  train['GarageCars'].isnull().sum())



train['GarageCars'].fillna(0,inplace=True)

comp['GarageCars'].fillna(0,inplace=True)



#print("Grouping on number of GarageCars:\n")

#print( train[['GarageCars','Id']].groupby('GarageCars').count())



#print("\n Data Type of field GarageArea : ",  train['GarageArea'].dtype)

#print("\n Record count for Missing  GarageArea : " ,  train['GarageArea'].isnull().sum())



train['GarageArea'].fillna(0,inplace=True)

comp['GarageArea'].fillna(0,inplace=True)



#print("Grouping on number of GarageArea:\n")

#print( train[['GarageArea','Id']].groupby('GarageArea').count())







#print("\n Data Type of field GarageYrBlt : ",  train['GarageYrBlt'].dtype)

#print("\n Record count for Missing  GarageYrBlt : " ,  train['GarageYrBlt'].isnull().sum())

#print("Grouping on number of GarageYrBlt:\n")



train['GarageYrBlt'].fillna(0,inplace=True)

train['GarageYrBlt']=train['GarageYrBlt'].astype(np.int64)



comp['GarageYrBlt'].fillna(0,inplace=True)

comp['GarageYrBlt']=comp['GarageYrBlt'].astype(np.int64)



#print("\n Data Type of field GarageYrBlt : ",  train['GarageYrBlt'].dtype)

#print( train[['GarageYrBlt','Id']].groupby('GarageYrBlt').count())



comp.loc[comp[comp['GarageYrBlt']==0].index, ['GarageYrBlt','GarageType']].count()


tmp1= train[ ['GarageCars','GarageArea','TotRmsAbvGrd','GrLivArea' , 'TotalBath','SalePrice']] 

pearson_corrrelation    = tmp1.corr(method="pearson");

print("\n Pearson correlation coefficient Between GarageCars,GarageArea TotRmsAbvGrd ,KitchenAbvGr, GrLivArea, BedroomAbvGr,TotalBath variables and SalePrice: \n");

print(pearson_corrrelation);



sns.pairplot(tmp1)



#train[ 'GarageType'].value_counts().plot(kind='bar')

#train[ 'GarageQual'].value_counts().plot(kind='bar')

#train[ 'GarageCond'].value_counts().plot(kind='bar')

#train[ 'GarageFinish'].value_counts().plot(kind='bar')

"""

PavedDrive : Paved driveway

Missing values replaced with 'N= No PavedDrive'



Majority of values  (Over 90%) are "Y" for this variable, so it is not good predictor variable

"""

print("\n Data Type of field PavedDrive : ",  train['PavedDrive'].dtype)

print("\n Record count for Missing  PavedDrive : " ,  train['PavedDrive'].isnull().sum())

print("\n Missing value is replaced with N, where N=No PavedDrive' : ")





train['PavedDrive'].fillna('N',inplace=True)

comp['PavedDrive'].fillna('N',inplace=True)



print("Train : Grouping on PavedDrive Quality :\n")

print( train[['PavedDrive','Id']].groupby(['PavedDrive']).count()['Id'])

train['PavedDrive'].value_counts().plot(kind='bar')

"""

WoodDeckSF : Wood deck area in square feet

Missing values replaced with 0



Majority of values  (Over 50%) are 0 for this variable, so it is not good predictor variable and need to convert into

category indicator variable eg Has_WoodDeck = 'Y' when WoodDeckSF !=0 otherwise 'N'  



"""



#print("\n Data Type of field WoodDeckSF : ",  train['WoodDeckSF'].dtype)

#print("\n Record count for Missing  WoodDeckSF : " ,  train['WoodDeckSF'].isnull().sum())

#print("\n Record count for Missing  WoodDeckSF=0 are : " ,  train[train['WoodDeckSF']==0].count()[0])

#print("\n Missing value is replaced with 0 : ")



train['WoodDeckSF'].fillna(0,inplace=True)

train['Has_Wooddeck'] = train['WoodDeckSF'].apply(lambda x: 'N' if x ==0 else 'Y')



#print("Grouping on WoodDeckSF  :\n")

#print( train[['WoodDeckSF','Id']].groupby(['WoodDeckSF']).count()['Id'])

#print(train['WoodDeckSF'].describe())





comp['WoodDeckSF'].fillna(0,inplace=True)

comp['Has_Wooddeck'] = comp['WoodDeckSF'].apply(lambda x: 'N' if x ==0 else 'Y')



print("\n For train dataset:")

print( train[['Has_Wooddeck','Id']].groupby(['Has_Wooddeck']).count()['Id'])

#sns.distplot(train['WoodDeckSF'])



"""

OpenPorchSF : Open porch area in square feet 

EnclosedPorch : Enclosed porch area in square feet

3SsnPorch : Three season porch area in square feet

ScreenPorch: Screen porch area in square feet



Missing values replaced with 0



We created variable Total_Poarch based on sum of all these poarch variable.

Its weakly correlated with SalePrice

To indicate the house has poarch or not we created categorical variable Has_Poarch which is 'Y' if

Total_Poarch !=0 otherwise 'N'

later we can drop all these variables: OpenPorchSF, EnclosedPorch , 3SsnPorch, ScreenPorch and Total_Poarch

and just keep Has_Poarch variable



"""



#print("\n Data Type of field OpenPorchSF : ",  train['OpenPorchSF'].dtype)

#print("\n Record count for Missing  OpenPorchSF : " ,  train['OpenPorchSF'].isnull().sum())

#print("\n Record count for Missing  OpenPorchSF=0 are : " ,  train[train['OpenPorchSF']==0].count()[0])

#print("\n Missing value is replaced with 0 : ")



train['OpenPorchSF'].fillna(0,inplace=True)



#print(train['OpenPorchSF'].describe())

#sns.distplot(train['OpenPorchSF'])



#print("\n Data Type of field EnclosedPorch : ",  train['EnclosedPorch'].dtype)

#print("\n Record count for Missing  EnclosedPorch : " ,  train['EnclosedPorch'].isnull().sum())

#print("\n Record count for Missing  EnclosedPorch=0 are : " ,  train[train['EnclosedPorch']==0].count()[0])

#print("\n Missing value is replaced with 0 : ")



train['EnclosedPorch'].fillna(0,inplace=True)



#print(train['EnclosedPorch'].describe())

#sns.distplot(train['EnclosedPorch'])



#print("\n Data Type of field 3SsnPorch : ",  train['3SsnPorch'].dtype)

#print("\n Record count for Missing  3SsnPorch : " ,  train['3SsnPorch'].isnull().sum())

#print("\n Record count for Missing  3SsnPorch=0 are : " ,  train[train['3SsnPorch']==0].count()[0])

#print("\n Missing value is replaced with 0 : ")



train['3SsnPorch'].fillna(0,inplace=True)





#print(train['3SsnPorch'].describe())

#sns.distplot(train['3SsnPorch'])



#print("\n Data Type of field ScreenPorch : ",  train['ScreenPorch'].dtype)

#print("\n Record count for Missing  ScreenPorch : " ,  train['ScreenPorch'].isnull().sum())

#print("\n Record count for Missing  ScreenPorch=0 are : " ,  train[train['ScreenPorch']==0].count()[0])

#print("\n Missing value is replaced with 0 : ")



train['ScreenPorch'].fillna(0,inplace=True)

                          

comp['OpenPorchSF'].fillna(0,inplace=True)

comp['EnclosedPorch'].fillna(0,inplace=True)

comp['3SsnPorch'].fillna(0,inplace=True)

comp['ScreenPorch'].fillna(0,inplace=True)



#print(train['ScreenPorch'].describe())

#sns.distplot(train['ScreenPorch'])



############

#train[['OpenPorchSF' , 'EnclosedPorch' , '3SsnPorch' , 'ScreenPorch']]



train['Total_Poarch']= train['OpenPorchSF']+train['EnclosedPorch']+train['3SsnPorch']+train['ScreenPorch']

train['Has_Poarch']=train['Total_Poarch'].apply(lambda x: 'Y' if x != 0 else 'N')



comp['Total_Poarch']= comp['OpenPorchSF']+comp['EnclosedPorch']+comp['3SsnPorch']+comp['ScreenPorch']

comp['Has_Poarch']=comp['Total_Poarch'].apply(lambda x: 'Y' if x != 0 else 'N')





#train[['OpenPorchSF' , 'EnclosedPorch' , '3SsnPorch' , 'ScreenPorch' , 'Total_Poarch', 'Has_Poarch']]



#print(train['Total_Poarch'].describe())

#print( train[['Has_Poarch','Id']].groupby(['Has_Poarch']).count()['Id'])

#sns.scatterplot(x='Total_Poarch', y='SalePrice', data=train)



#tmp1= train[ ['Total_Poarch','GarageArea','TotRmsAbvGrd','GrLivArea' , 'TotalBath','SalePrice']] 

#pearson_corrrelation    = tmp1.corr(method="pearson");

#print("\n Pearson correlation coefficient Between Total_Poarch ,GarageArea TotRmsAbvGrd ,KitchenAbvGr, GrLivArea, BedroomAbvGr,TotalBath variables and SalePrice: \n");

#print(pearson_corrrelation);



sns.pairplot(tmp1)

"""

Fence : Fence quality



Over 80% of values are missing which we considered as 'No Fence' for this category variable



"""



#Fence variable



#print(" Fence variable detail: \n")

#print( "\n Datatype of column Fence : ", train['Fence'].dtype  )

#print( train['Fence'].describe()  )



#print("\n Missing Counts for column Fence :", train['Fence'].isnull().sum())

#print("\n Grouping of the values for column Fence: \n")

#print( train.groupby('Fence').count()['Id']  )



#Converting Fence value ofMissing /  NA to “No Fence”



train['Fence'].fillna('No Fence', inplace=True)

comp['Fence'].fillna('No Fence', inplace=True)





#print("\n Grouping of the values for column Fence after filling  for Missing with 'No Fence' : \n")

#print( train.groupby('Fence').count()['Id']  )



train['Fence'].value_counts().plot(kind='bar')

"""

MiscFeature : Miscellaneous feature not covered in other categories

MiscVal : $Value of miscellaneous feature



#Converting MiscFeature value of Missing / NA to “No Feature”



# MiscFeature and MiscVal are related as  MiscVall should be non-zero when a non missing/NA value exists in MiscFeature 

   and should be zero when MiscFeature is Missing / NA.

I found two observation where there is MiscFeature but MiscVal is 0

         MiscFeature  MiscVal

873         Othr        0

1200        Shed        0



  It appears that MiscVal is > 0 for houses that have MiscFeature  and zero for houses that have no MiscFeature.

  It’s clear that the MiscFeature value of NA indicates a condition where no misc feature exists. 

  We don't have such records



Note:

As Majority of   MiscFeature are missing / “No Feature” with corresponding MiscVal=0

both these variable can be dropped.



"""





#print("\n  Datatype of field MiscFeature is: ",train['MiscFeature'].dtype)

#print("\n  count of Missing/NA for field MiscFeature is: ",train['MiscFeature'].isnull().sum())

#print("\n Grouping of field MiscFeature : \n ")

#print(   train.groupby('MiscFeature').count()['Id'])



Miss_Miscval =  train.loc[ train[train['MiscFeature'].notnull()].index    , [ 'MiscFeature','MiscVal' ]]

Miss_Miscval=train.loc[  Miss_Miscval[Miss_Miscval['MiscVal']==0].index , [ 'MiscFeature','MiscVal'] ]



Miss_Miscval_c =  comp.loc[ comp[comp['MiscFeature'].notnull()].index    , [ 'MiscFeature','MiscVal' ]]

Miss_Miscval_c= comp.loc[  Miss_Miscval_c[Miss_Miscval_c['MiscVal']==0].index , [ 'MiscFeature','MiscVal'] ]





#print(" \n List of not null MiscFeatures and corresponding Missing/NA/0  MiscVal (needs to be updated) : \n")

#print(Miss_Miscval)



Miss_MiscFeature =  train.loc[ train[train['MiscVal']!=0].index    , [ 'MiscFeature','MiscVal' ]]

Miss_MiscFeature =  train.loc[  Miss_MiscFeature[Miss_MiscFeature['MiscFeature']== np.NaN].index , [ 'MiscFeature','MiscVal'] ]



Miss_MiscFeature_c =  comp.loc[ comp[comp['MiscVal']!=0].index    , [ 'MiscFeature','MiscVal' ]]

Miss_MiscFeature_c =  comp.loc[  Miss_MiscFeature_c[Miss_MiscFeature_c['MiscFeature']== np.NaN].index , [ 'MiscFeature','MiscVal'] ]



#print(" \n List of not null MiscVal and corresponding Missing/NA MiscFeature : \n")

#print(Miss_MiscFeature)





#print("\n  Datatype of field MiscVal is: ",train['MiscVal'].dtype)

#print("\n  count of Missing/NA for field MiscVal is: ",train['MiscVal'].isnull().sum())

#print(train['MiscVal'].describe())



#After updating value with mean value for records with MiscFeatures available



Miscval_avg=train.loc[train[train['MiscVal']!=0].index , ['MiscVal']].mean().astype(int) 

indices=Miss_Miscval.index.values.tolist()



Miscval_avg_c = comp.loc[comp[comp['MiscVal']!=0].index , ['MiscVal']].mean().astype(int) 

indices_c=Miss_Miscval_c.index.values.tolist()



#print(indices)

train.loc[indices,'MiscVal']=Miscval_avg[0]

comp.loc[indices_c,'MiscVal']=Miscval_avg_c[0]



#print(train.loc[Miss_Miscval.index.values.tolist(),['MiscVal']])    

#print(train.loc[indices,:])



#print("\n After updating value with mean value for records with MiscFeatures available\n")

#print(train['MiscVal'].describe())



#print("\n After updating missing MiscFeature to No Feature :\n")



train['MiscFeature'].fillna('No Feature', inplace=True)

comp['MiscFeature'].fillna('No Feature', inplace=True)





#print(train.groupby('MiscFeature').count()['Id'])





"""

OverallQual : Overall material and finish quality

OverallCond : Overall condition rating



There are no missing values in these discrete variables. Correlation of 'OverallQual' with 'SalePrice' is strong where as

'OverallCond' variable's coorelation with both 'SalePrice' and 'OverallQual' is weak.



So we decided to drop variable 'OverallCond' and keep variable 'OverallQual'

"""



#print("\n Data Type of field OverallQual : ",  train['OverallQual'].dtype)

#print("\n Record count for Missing  OverallQual : " ,  train['OverallQual'].isnull().sum())



train['OverallQual'].fillna(0,inplace=True)



#print("\n  Grouping of field OverallQual : ")

#print(train.groupby('OverallQual').count()['Id'])





#print("\n Data Type of field OverallCond : ",  train['OverallCond'].dtype)

#print("\n Record count for Missing  OverallCond : " ,  train['OverallCond'].isnull().sum())



train['OverallCond'].fillna(0,inplace=True)





#print("\n  Grouping of field OverallCond : ")

#print(train.groupby('OverallCond').count()['Id'])



#tmp1= train[ ['OverallQual','OverallCond','Total_Poarch','GarageArea','TotRmsAbvGrd','GrLivArea' , 'TotalBath','SalePrice']] 

#pearson_corrrelation    = tmp1.corr(method="pearson");

#print("\n Pearson correlation coefficient Between OverallQual,OverallCond,Total_Poarch ,GarageArea TotRmsAbvGrd ,KitchenAbvGr, GrLivArea, BedroomAbvGr,TotalBath variables and SalePrice: \n");

#print(pearson_corrrelation);



#sns.scatterplot(x='OverallCond', y='OverallQual', data=train)



#sns.pairplot(tmp1)



comp['OverallQual'].fillna(0,inplace=True)

comp['OverallCond'].fillna(0,inplace=True)

"""

YearRemodAdd : Remodel date



This discrete variable has moderate correlation with SalePrice, initially we will keep it for modeling.

"""





#print("\n Data Type of field YearRemodAdd : ",  train['YearRemodAdd'].dtype)

#print("\n Record count for Missing  YearRemodAdd : " ,  train['YearRemodAdd'].isnull().sum())



train['YearRemodAdd'].fillna(0,inplace=True)

comp['YearRemodAdd'].fillna(0,inplace=True)



#print("\n  Grouping of field YearRemodAdd : ")

#print(train.groupby('YearRemodAdd').count()['Id'])





#tmp1= train[ ['YearRemodAdd','OverallQual','Total_Poarch','GarageArea','TotRmsAbvGrd','GrLivArea' , 'TotalBath','SalePrice']] 

#pearson_corrrelation    = tmp1.corr(method="pearson");

#print("\n Pearson correlation coefficient Between OverallQual,OverallCond,Total_Poarch ,GarageArea TotRmsAbvGrd ,KitchenAbvGr, GrLivArea, BedroomAbvGr,TotalBath variables and SalePrice: \n");

#print(pearson_corrrelation);





#sns.scatterplot(x='YearRemodAdd', y='SalePrice', data=train)

#sns.pairplot(tmp1)
"""

   LotArea : Lot size in square feet



There is no missing value, if it comes should be replaced with 0/mean

slightly right skewed distribution



It has weak positive correlation with 'SalePrice' and 'GrLivArea'





"""

#print("\n Data Type of field LotArea : ",  train['LotArea'].dtype)

#print("\n Record count for Missing  LotArea : " ,  train['LotArea'].isnull().sum())



train['LotArea'].fillna(0,inplace=True)

comp['LotArea'].fillna(0,inplace=True)



#print("\n  Grouping of field LotArea : ")

#print(train.groupby('LotArea').count()['Id'])  #too many different values...



LotArea_p1=train['LotArea'].quantile(.01)

LotArea_p99=train['LotArea'].quantile(.99)

LotArea_p25=train['LotArea'].quantile(.25)

LotArea_p75=train['LotArea'].quantile(.75)

LotArea_IQR=np.abs(LotArea_p75-LotArea_p25)

LotArea_lb= LotArea_p25 - (1.5 * LotArea_IQR)

LotArea_ub= LotArea_p75 + (1.5 * LotArea_IQR)



#print(train['LotArea'].describe())

#print( '\n LotArea_p1 : ', LotArea_p1)

#print( '\n LotArea_p25 : ', LotArea_p25)

#print( '\n LotArea_p75 : ', LotArea_p75)

#print( '\n LotArea_p99 : ', LotArea_p99)

#print( '\n LotArea_IQR : ', LotArea_IQR)



print( '\n LotArea_lb : ', LotArea_lb)

print( '\n LotArea_ub : ', LotArea_ub)



#print( '\n Count of outliers in LotArea: ', train.loc[ train[ (train['LotArea'] < LotArea_lb) | (train['LotArea'] > LotArea_ub )].index, ['Id','LotArea']].count()['Id']  )

      

#print (train.loc[ train[ (train['LotArea'] < LotArea_lb) | (train['LotArea'] > LotArea_ub )].index, ['Id','LotArea']]   )



#sns.distplot(train['LotArea'])



#tmp1= train[ ['LotArea','YearRemodAdd','OverallQual','Total_Poarch','GarageArea','TotRmsAbvGrd','GrLivArea' , 'TotalBath','SalePrice']] 

#pearson_corrrelation    = tmp1.corr(method="pearson");

#print("\n Pearson correlation coefficient Between LotArea, OverallQual,Total_Poarch ,GarageArea TotRmsAbvGrd ,KitchenAbvGr, GrLivArea, BedroomAbvGr,TotalBath variables and SalePrice: \n");

#print(pearson_corrrelation);





#sns.scatterplot(x='YearRemodAdd', y='SalePrice', data=train)

#sns.pairplot(tmp1)


"""

SaleType : Type of sale





For the sale type variable, we decided to trim the leading and trailing space of the categories. 

Then, after looking at the bar plot and boxplot of sale price by sale type

, we combined the SaleType levels to 'WD' -> warranty deeds, new sales (first owner)

, and all 'Other' sale types.





"""



#print("\n Data Type of field SaleType : ",  train['SaleType'].dtype)

#print("\n Record count for Missing  SaleType : " ,  train['SaleType'].isnull().sum())

#print(train['SaleType'].describe())



train['SaleType'].fillna('No SaleType', inplace=True)

train['SaleType'] = train['SaleType'].str.strip()





#print(train.groupby('SaleType').count()['Id'])



train['SaleType'] =train['SaleType'] .apply(lambda x : x if x in('WD','New') else 'Other')



comp['SaleType'].fillna('No SaleType', inplace=True)

comp['SaleType'] = comp['SaleType'].str.strip()

comp['SaleType'] =comp['SaleType'] .apply(lambda x : x if x in('WD','New') else 'Other')



#print("\n After updating SaleType \n")

#print(train.groupby('SaleType').count()['Id'])


"""



SaleCondition : Condition of sale





For the SaleCondition variable, we decided to trim the leading and trailing space. 

Then, after looking at the bar plot and boxplot of SalePrice by SaleType and SaleCondition

, we combined the SaleCondition into Normal','Abnorml' ,'Partial' and all 'Other' SaleCondition.





"""



#print("\n Data Type of field SaleCondition : ",  train['SaleCondition'].dtype)

#print("\n Record count for Missing  SaleCondition : " ,  train['SaleCondition'].isnull().sum())

#print(train['SaleCondition'].describe())



train['SaleCondition'].fillna('No SaleCondition', inplace=True)

train['SaleCondition'] = train['SaleCondition'].str.strip()







#print(train.groupby('SaleCondition').count()['Id'])



train['SaleCondition'] =train['SaleCondition'] .apply(lambda x : x if x in('Normal','Abnorml' ,'Partial') else 'Other')





comp['SaleCondition'].fillna('No SaleCondition', inplace=True)

comp['SaleCondition'] = comp['SaleCondition'].str.strip()

comp['SaleCondition'] =comp['SaleCondition'] .apply(lambda x : x if x in('Normal','Abnorml' ,'Partial') else 'Other')



print("\n After updating SaleCondition in train dataset \n")

print(train.groupby('SaleCondition').count()['Id'])
"""

Target variable:



SalePrice : the property's sale price in dollars. This is the target variable that you're trying to predict.



No missing values in SalePrice

Distribution plot suggest there are outliers in SalePrice and needs to be dropped



"""

#print("\n Data Type of field SalePrice : ",  train['SalePrice'].dtype)

#print("\n Record count for Missing  SalePrice : " ,  train['SalePrice'].isnull().sum())

#print(train['SalePrice'].describe())



# Calculating Outliers:

SalePrice_p1=train['SalePrice'].quantile(.01)

SalePrice_p99=train['SalePrice'].quantile(.99)

SalePrice_p25=train['SalePrice'].quantile(.25)

SalePrice_p75=train['SalePrice'].quantile(.75)

SalePrice_IQR=np.abs(SalePrice_p75-SalePrice_p25)

SalePrice_lb= SalePrice_p25 - (1.5 * SalePrice_IQR)

SalePrice_ub= SalePrice_p75 + (1.5* SalePrice_IQR)



#print(train['SalePrice'].describe())

#print( '\n SalePrice_p1 : ', SalePrice_p1)

#print( '\n SalePrice_p25 : ', SalePrice_p25)

#print( '\n SalePrice_p75 : ', SalePrice_p75)

#print( '\n SalePrice_p99 : ', SalePrice_p99)

#print( '\n SalePrice_IQR : ', SalePrice_IQR)

print( '\n SalePrice_lb : ', SalePrice_lb)

print( '\n SalePrice_ub : ', SalePrice_ub)



#print( '\n SalePrice Outlier drop count: ', train.loc[ train[ (train['SalePrice'] < SalePrice_lb) | (train['SalePrice'] > SalePrice_ub )].index, :].count()['Id'])

 

sns.distplot(train['SalePrice'])

#Creating list of outliers records to review later , it may overlap



train_SalePrice_Outliers=train.loc[ train[ (train['SalePrice'] < SalePrice_lb) | (train['SalePrice'] > SalePrice_ub )].index, :]

train_LotArea_Outliers=train.loc[ train[ (train['LotArea'] < LotArea_lb) | (train['LotArea'] > LotArea_ub )].index, :]

train_GrLivArea_Outliers=train.loc[ train[ (train['GrLivArea'] < GrLivArea_lb) | (train['GrLivArea'] > GrLivArea_ub ) ].index, :] 



print("\n There can be common outliers:")

print('train_SalePrice_Outliers :',train_SalePrice_Outliers.count()['Id'])         

print('train_LotArea_Outliers :',train_LotArea_Outliers.count()['Id'])         

print('train_GrLivArea_Outliers :',train_GrLivArea_Outliers.count()['Id'])           

                                           
#Dropping outliers records (Should be done only from train dataset)



# Transform some variables as required using log_transformation and drop original variables



print("Dropping all outliers for SalePrice, GrLivArea and  LotArea :")

print('Train count before start deleting outliers : ' , train.count()['Id'])



train = train.loc[ train[ (train['SalePrice'] >= SalePrice_lb) & (train['SalePrice'] <= SalePrice_ub )].index, :]

train=train.loc[ train[ (train['LotArea'] >= LotArea_lb) & (train['LotArea'] <= LotArea_ub )].index, :]

train=train.loc[ train[ (train['GrLivArea'] >= GrLivArea_lb) & (train['GrLivArea'] <= GrLivArea_ub ) ].index, :] 



print('Train count after  deleting outliers  : ' , train.count()['Id'])





# convert them as str type:   'OverallQual','YearRemodAdd', 'MasVnrType' ,'Has_Bsmt', 'Has_FirePlace'



train['OverallQual']=  train['OverallQual'].astype(str)

train['MasVnrType']=  train['MasVnrType'].astype(str)

train['Has_Bsmt']=  train['Has_Bsmt'].astype(str)

train['Has_FirePlace']=  train['Has_FirePlace'].astype(str)

train['YearRemodAdd']=  train['YearRemodAdd'].astype(str)



comp['OverallQual']=  comp['OverallQual'].astype(str)

comp['MasVnrType']=  comp['MasVnrType'].astype(str)

comp['Has_Bsmt']=  comp['Has_Bsmt'].astype(str)

comp['Has_FirePlace']=  comp['Has_FirePlace'].astype(str)

comp['YearRemodAdd']=  comp['YearRemodAdd'].astype(str)





#generate log_field which are not linear in distribution



train['log_LotArea'] = np.log(train['LotArea'])

train['log_GrLivArea'] = np.log(train['GrLivArea'])

train['log_GarageArea']=train['GarageArea']

train['log_TotalBsmtSF'] = train['TotalBsmtSF']

train['log_LotFrontage'] = train['LotFrontage']

train.loc[ train[ train['log_GarageArea'] != 0].index, 'log_GarageArea']  = np.log(train['log_GarageArea'])

train.loc[ train[ train['log_TotalBsmtSF'] != 0].index, 'log_TotalBsmtSF'] = np.log(train['log_TotalBsmtSF'])

train.loc[ train[ train['log_LotFrontage'] != 0].index, 'log_LotFrontage']  = np.log(train['log_LotFrontage'])

train['log_SalePrice'] = np.log(train['SalePrice'])





comp['log_LotArea'] = np.log(comp['LotArea'])

comp['log_GrLivArea'] = np.log(comp['GrLivArea'])

comp['log_GarageArea']=comp['GarageArea']

comp['log_TotalBsmtSF'] = comp['TotalBsmtSF']

comp['log_LotFrontage'] = comp['LotFrontage']

comp.loc[ comp[ comp['log_GarageArea'] != 0].index, 'log_GarageArea']  = np.log(comp['log_GarageArea'])

comp.loc[ comp[ comp['log_TotalBsmtSF'] != 0].index, 'log_TotalBsmtSF'] = np.log(comp['log_TotalBsmtSF'])

comp.loc[ comp[ comp['log_LotFrontage'] != 0].index, 'log_LotFrontage']  = np.log(comp['log_LotFrontage'])

# drop fields for which log transformation is done



train.drop(['LotArea'],axis=1,inplace=True)

train.drop(['GrLivArea'],axis=1,inplace=True)

train.drop(['GarageArea'],axis=1,inplace=True)

train.drop(['TotalBsmtSF'],axis=1,inplace=True)

train.drop(['LotFrontage'],axis=1,inplace=True)

train.drop(['SalePrice'],axis=1,inplace=True)



# drop fields for which log transformation is done



comp.drop(['LotArea'],axis=1,inplace=True)

comp.drop(['GrLivArea'],axis=1,inplace=True)

comp.drop(['GarageArea'],axis=1,inplace=True)

comp.drop(['TotalBsmtSF'],axis=1,inplace=True)

comp.drop(['LotFrontage'],axis=1,inplace=True)







train_non_obj=[]

train_obj=[]

for x in train.columns.values:

    if (train[x].dtype=='object'):

        train_obj.append(x)

    else:

        train_non_obj.append(x)



print('\n train_obj : \n',train_obj)

print('\n train_non_obj: \n',train_non_obj)



comp_non_obj=[]

comp_obj=[]

for x in comp.columns.values:

    if (comp[x].dtype=='object'):

        comp_obj.append(x)

    else:

        comp_non_obj.append(x)



print('\n comp_obj : \n',comp_obj)

print('\n comp_non_obj: \n',comp_non_obj)

#creating drop column list



"""

print(list(train.columns.values))



Based on above analysis, if the field has high number of missing values or same type of values,

having multicollinearity with other input variables, or very low colinearity with target variable or high vls

we drop those variables.



"""



       

#Transformed columns:    'LotFrontage',  'LotArea' , 'GrLivArea', 'GarageArea', 'TotalBsmtSF' , 'SalePrice' ]



drop_column_list=['Street', 'Alley', 'LandContour', 'Utilities', 'LandSlope','Condition1', 'Condition2',

                 'YearBuilt', 'RoofMatl', 'MasVnrArea', 'ExterCond', 'BsmtCond', 'BsmtFinType2', 'Heating',

                  'CentralAir', 'Electrical', 'Functional', 'PoolArea', 'PoolQC', 'Fireplaces', 'FireplaceQu',

                  'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',

                  'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',

                  'TotRmsAbvGrd', 'GarageQual', 'GarageCond', 'GarageYrBlt', 'GarageCars',  'PavedDrive',

                  'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'Total_Poarch',

                  'MoSold', 'YrSold', 'MiscFeature', 'MiscVal',  'OverallCond' ,'log_LotFrontage' 

                  , 'MSSubClass' , 'Exterior1st', 'Exterior2nd' , 'MasVnrType', 'HouseStyle'

                 ,'GarageType' , 'GarageFinish', 'Has_Bsmt' ,'BsmtQual', 'BsmtFinType1'

                 ]





keep_column_list=[x  for x in train.columns.values if x not in drop_column_list]

comp_keep_column_list=[x  for x in keep_column_list if x != 'log_SalePrice']



print("\n drop_column_list : \n" , drop_column_list)

       

print("\n keep_column_list : \n" , keep_column_list)

       

print("\n comp_keep_column_list : \n" , comp_keep_column_list)

       

train=train[keep_column_list]

comp=comp[comp_keep_column_list ]



pearson_corrrelation    = train[keep_column_list].corr(method="pearson");

print("\n Pearson correlation coefficient : \n");

print(pearson_corrrelation);



pearson_corrrelation    = comp[comp_keep_column_list].corr(method="pearson");

print("\n Pearson correlation coefficient : \n");

print(pearson_corrrelation);

#Preprocessing is almost complete except dropping Id field

#lets compare column list for train and comp after generating dummies..., divide train into inputs and targets



k_train1=train.copy()

c_comp1=comp.copy()



#k_train1.columns.values



k_train1_Id_log_SalePrice = k_train1[['Id','log_SalePrice' ]]

c_comp1_Id = c_comp1[['Id']]



k_train1.drop(['Id'], axis=1, inplace=True)

c_comp1.drop(['Id'], axis=1, inplace=True)



tmp1=list(c_comp1.columns.values)

tmp1.append('log_SalePrice')

print(k_train1.columns.values)

print(c_comp1.columns.values)



print( list(k_train1.columns.values)==  tmp1) #  i.e. the only diff is log_SalePrice



# Data With Dummies (dwd)



#train

dwd = pd.get_dummies(k_train1, drop_first=True)

targets=dwd['log_SalePrice']

inputs=dwd[[x for x in dwd.columns.values if x not in ['log_SalePrice'] ]]



#compitition



cdwd = pd.get_dummies(c_comp1, drop_first=True)

cinputs=cdwd

comp_col_not_in_train = [x for x in cinputs.columns.values if x not in (inputs.columns.values)]

train_col_not_in_comp = [x for x in inputs.columns.values if x not in (cinputs.columns.values)]



print("\n comp col Not in train with dummies\n")

print(comp_col_not_in_train)



print("\n train col Not in comp with dummies\n")

print(train_col_not_in_comp)



add_col_drop=train_col_not_in_comp + comp_col_not_in_train

print("\n Total additional col to drop: ")

print(add_col_drop)



for x in add_col_drop:

    if x in inputs.columns:

        inputs.drop(x,axis=1, inplace=True)

    if x in cinputs.columns:

        cinputs.drop(x,axis=1, inplace=True)

 

print(   list(inputs.columns.values) ==  list(cinputs.columns.values) )
#Lets generate file with column/ variable name  of model inputs that should present in any new data 

# which will run through the model we generate





f= open("/kaggle/working/inputs_ColList_with_Dummies_4_modelscoring.txt","w+")

for i in inputs.columns.values:

     f.write(i)

     f.write("\n")

f.close()

print("OK")
#making sure the newdata (test/competition data) have same columns according to model we are creating



print(inputs.shape)

print(cinputs.shape)



#print(inputs.index)

#print(cinputs.index)
#Scale the data

scaler=StandardScaler(copy=True,with_mean=True, with_std=True)

scaler.fit(inputs)

inputs_scaled=scaler.transform(inputs)



#split our train dataset data into training and testing

x_train ,x_test , y_train , y_test = train_test_split(inputs_scaled, targets,test_size=0.2,random_state=365)



#create regression



reg=LinearRegression()

reg.fit(x_train, y_train)

y_hat = reg.predict(x_train)



#First check :our model should pass 45 degree line, i.e. points should concentrated around it



plt.scatter(y_train,y_hat);

plt.xlabel('Targets (y_train)', size=18)

plt.ylabel('Prediction (y_hat)', size=18)

plt.show()

#Model score  --> 0.9171299431368138   Approx(92% score on training)



print(reg.score(x_train,y_train))  #Model score: 0.9171299431368138  during training

print(reg.intercept_)

print(reg.coef_)
#create summary table of features and weights



reg_summary=pd.DataFrame(inputs.columns.values,columns=['Features'])

reg_summary['Weights']=reg.coef_

print(reg_summary)
#testing: (test data generated from train dataset)





y_hat_test=reg.predict(x_test)





#Plot the test targets against the predicted targets and see if they resembles 45 degree line

#What can be observed that for lower mid to higher pricing they concentrate around 4 degree line (good to predict)

# however for very small price bit scattered

plt.scatter(y_test, y_hat_test,alpha=0.2)

plt.xlabel('Targets (y_test)', size=18)

plt.ylabel('Predictions (y_hat_test)', size=18)

plt.xlim(8,14)

plt.ylim(8,14)

plt.show()



#create test_prediction data frame



dfpf = pd.DataFrame(np.exp(y_hat_test) , columns=['Predictions'])

dfpf.head()

#There will be lot of missing values and there fore to avoid it as next step you need to reset index

y_test=y_test.reset_index(drop=True)



dfpf['Target']= np.exp(y_test)

dfpf.head(10)

#Lower the error means better explanation power



dfpf['Residual']=dfpf['Target'] - dfpf['Predictions']

dfpf['Diff%'] = np.absolute(dfpf['Residual']/dfpf['Target'] * 100)

dfpf.describe()

pd.options.display.max_rows=None

pd.set_option('display.float_format', lambda x : '%.2f'%x)

dfpf.sort_values(by=['Diff%'])    #(by=['Target'])
# save the model

dump(reg, open('/kaggle/working/House_Price_Prediction_model.pkl', 'wb'))

# save the scaler

dump(scaler, open('/kaggle/working/House_Price_Prediction_scaler.pkl', 'wb'))

print("\n model is saved for later use \n")    
#Now load the model and scaler from pickle file our compitition data (test file without SalePrice) to predict SalePrice

# load the model

reg_loaded = load(open('/kaggle/working/House_Price_Prediction_model.pkl', 'rb'))

# load the scaler



scaler_loaded = load(open('/kaggle/working/House_Price_Prediction_scaler.pkl', 'rb'))



print("\n now model and scaler are loaded :\n")
# fit Competition data (Already preprocessed, dummy var gen ) into scaler_loaded object



scaler_loaded.fit(cinputs)



#scale the inputs for the Competition 



cinputs_scaled=scaler_loaded.transform(cinputs)

print("\n scaled input is generated")



#predict SalePrice for Competition data and place it in dataframe with ID variable for competition data



cy_hat = reg_loaded.predict(cinputs_scaled)



c_comp1_Id['sp']=cy_hat



print("\n predicted log_SalePrice for Competition data is generated and place it in dataframe with ID variable for competition data")

# This SalePrice prediction is for log transformed SalePrice variable so we need to take its exponent



#print(np.exp(12.07))

#print(c_comp1_Id.shape)

if ('SalePrice' in c_comp1_Id.columns.values ):

        c_comp1_Id.drop(['SalePrice'], axis=1, inplace=True)

if ('sp' not  in c_comp1_Id.columns.values ):

        c_comp1_Id['sp']=cy_hat        

c_comp1_Id['SalePrice']= np.exp(c_comp1_Id['sp'])

c_comp1_Id.drop(['sp'], axis=1, inplace=True)

c_comp1_Id['SalePrice']=c_comp1_Id['SalePrice'].apply(lambda x:round(x,2))

print("\n final SalePrice rounded to two decimal places is generated and ready to export")

print(c_comp1_Id)



c_comp1_Id.to_csv("/kaggle/working/Predicted_SalePrice_submission.csv", index=False)

print("\n Predicted_SalePrice.csv file is exported")

print(c_comp1_Id.describe())

sns.distplot(c_comp1_Id['SalePrice'])

plt.title("Predicted SalesPrice distribution")

print("Complete")