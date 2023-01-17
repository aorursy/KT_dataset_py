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
#panel data analysis - pandas

import pandas as pd



#numpy for operations related to numpy arrays/series

import numpy as np



#matplotlib & seaborn for visualisations

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline  

#the above line of code is a magic function, enables us to display plots within our notebooks just below the code.



#maths, stats and stuffs

from scipy import stats
train_df=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test_df=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
#.columns method gives us name of all the columns in our dataframe

train_df.columns
#.dtypes returns a series of all columns and their respective datatype

train_df.dtypes
test_df.columns
test_df.dtypes
#dataframe.info() is like one stop destination for all info on metadata

train_df.info()
test_df.info()
# by default it displays the first 5 records, any other integer can be specified within the parens as I did, giving 6.

train_df.head(6)
test_df.head(6)
#describe gives us a stats about every NUMERICAL column.

train_df.describe()
#using select_dtypes to select specified data type

train_df.select_dtypes('object').describe()   #this code gives a short statistical summary of categorical data. # object means string type here
# unique () used to get all distinct values from columns

train_df.LotShape.unique()
# value_counts() is even better, gives distinct values with their frequencies.

train_df.LotShape.value_counts()
num_col=train_df.select_dtypes(exclude='object')
for i in num_col.columns:

    num_col[i].plot.hist(bins=40,color=('r'))

    plt.xlabel(i)

    plt.show()
# the .corr() method helps compute correlation of columns with each other, it excludes nulls automatically

corrmap=train_df.corr()
# .corr(), returns a correlation matrix, which is displayed below, entries are correlation values

corrmap
# below code is to get those features which have correlation greater than 0.5 with Target-SalePrice



best_corrd=corrmap.index[abs(corrmap['SalePrice'])>0.5] #-ve corr. value means they are correlated but inversely, still we need 'em, hence abs()

print(best_corrd)
plt.figure(figsize=(12,12))

sns.heatmap(train_df[best_corrd].corr(),annot=True,cmap='RdYlBu')
# all in one plot - the seaborn pairplot

col= ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(train_df[col],height=1.9,aspect=0.99,diag_kind='kde')  # for each facet -> width=aspect*height & height is in inches
f,ax=plt.subplots(4,2,figsize=(15,10))    #method of matplot, allows to plot 2 or more graphs in same figure



#plotting OverallQual with SalePrice.. here both lineplot and barplot of OverallQual are plotted on same graph

sns.lineplot(train_df['OverallQual'],train_df['SalePrice'],ax=ax[0,0])

sns.barplot(train_df['OverallQual'],train_df['SalePrice'],ax=ax[0,0])



#plotting TotalBsmtSF with SalePrice..

sns.scatterplot(train_df['TotalBsmtSF'],train_df['SalePrice'],ax=ax[0,1])



#plotting 1stFlrSF with SalePrice..

sns.scatterplot(train_df['1stFlrSF'],train_df['SalePrice'],marker="*",ax=ax[1,0])    #marker is the shape of the points on scatterplot



#plotting GrLivArea with SalePrice..

sns.scatterplot(train_df['GrLivArea'],train_df['SalePrice'],marker="+",ax=ax[1,1])



#plotting GarageCars with SalePrice..

sns.lineplot(train_df['GarageCars'],train_df['SalePrice'],ax=ax[2,0])

sns.barplot(train_df['GarageCars'],train_df['SalePrice'],ax=ax[2,0])



#plotting GarageArea with SalePrice..

sns.scatterplot(train_df['GarageArea'],train_df['SalePrice'],ax=ax[2,1])



#plotting TotRmsAbvGrd with SalePrice..

sns.barplot(train_df['TotRmsAbvGrd'],train_df['SalePrice'],ax=ax[3,0])



#plotting YearBuilt with SalePrice..

sns.lineplot(train_df['YearBuilt'],train_df['SalePrice'],ax=ax[3,1])



plt.tight_layout()   #this automatically adjusts the placement of the plots in the figure area, without this, the figure labels were overlapping 
train_df.drop(train_df[train_df.GrLivArea>4000].index, inplace = True)

train_df.drop(train_df[train_df.TotalBsmtSF>3000].index, inplace = True)

train_df.drop(train_df[train_df.GrLivArea>4000].index, inplace = True)

train_df.drop(train_df[train_df.YearBuilt<1900].index, inplace = True)  # we see peak in sale price for few houses pre 1900, hence OUTLIER
#post outlier treatment

sns.scatterplot(train_df.GrLivArea,train_df.SalePrice)
#post outlier treatment

sns.scatterplot(train_df.TotalBsmtSF,train_df.SalePrice)
#extracting our target into separate var

y_train=train_df.SalePrice

train_df.drop(columns=['SalePrice'],inplace=True)
train_df.columns==test_df.columns # to show train and test datasets have similar columns, so concat them and treat nulls
df_merged = pd.concat([train_df, test_df], axis = 0) #axis=0 to concat along rows ; axis=1 is for columns

df_merged.shape
#some vars, though have numerical values but are actually categorical, so convert them

df_merged[['MSSubClass', 'OverallQual', 'OverallCond', 'MoSold', 'YrSold']] = df_merged[['MSSubClass', 'OverallQual', 'OverallCond', 'MoSold', 'YrSold']].astype('object')

df_merged.dtypes.value_counts()
#columns with missing values

missing_columns = df_merged.columns[df_merged.isnull().any()]

print(missing_columns)

print(len(missing_columns))
#to find how many nulls

df_merged[missing_columns].isnull().sum().sort_values(ascending=False)
# impute by "NONE", wherever NaN means absence of that feature in the house



none_imputer = df_merged[['PoolQC','MiscFeature','Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageCond','GarageFinish','GarageQual','BsmtFinType2','BsmtExposure','BsmtQual','BsmtCond','BsmtFinType1','MasVnrType']]

for i in none_imputer.columns:

    df_merged[i].fillna('None', inplace = True)
# filling nulls in categorical vars with mode



mode_imputer =  df_merged[['Electrical', 'MSZoning','Utilities','Exterior1st','Exterior2nd','KitchenQual','Functional', 'SaleType']]

for i in mode_imputer.columns:

    df_merged[i].fillna(df_merged[i].mode()[0], inplace = True)  #.mode()[0] because if var. is multimodal, then take the first one
# dealing with numericals, filling with median (robust to outliers)



median_imputer = df_merged[['BsmtFullBath','BsmtHalfBath', 'GarageCars', 'MasVnrArea', 'GarageYrBlt', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageArea','LotFrontage']]

for i in median_imputer.columns:

    df_merged[i].fillna(df_merged[i].median(), inplace = True)
print(len(df_merged.columns))

print(df_merged.isnull().any().value_counts())  #checking if any nulls in any columns..
#checking our target variable

sns.distplot(y_train)

print('SalePrice skew :',stats.skew(y_train))
# applying log trnsm on saleprice

y_train=np.log1p(y_train)      #remember we imported numpy as np

sns.distplot(y_train)

print('SalePrice skew post transformation:',stats.skew(y_train))
#checking skewness of other variables

skewed = pd.DataFrame(data = df_merged.select_dtypes(exclude='object').skew(), columns=['Skew']).sort_values(by='Skew',ascending=False)

plt.figure(figsize=(6,13))

sns.barplot(y=skewed.index,x='Skew',data=skewed)
#filtering numeric vars

df_merged_num = df_merged.select_dtypes(exclude='object')

df_merged_num.head(2)
# transforming vars where skew is high

df_trnsfmed=np.log1p(df_merged_num[df_merged_num.skew()[df_merged_num.skew()>0.5].index])



#other vars which have skew<0.5

df_untrnsfmd=df_merged_num[df_merged_num.skew()[df_merged_num.skew()<0.5].index]



#concat them

df_allnums=pd.concat([df_trnsfmed,df_untrnsfmd],axis=1)  #axis=1 coz conact along columns



df_merged_num.update(df_allnums)

df_merged_num.shape
#filtering only those which are categorical in type

df_merged_cat=df_merged.select_dtypes(exclude=['int64','float64'])

df_merged_cat.head()
#encoding the vars using pandas get dummies, get_dummies encodes all cat. vars.

df_dummy_cat=pd.get_dummies(df_merged_cat)
#final merging of normalised numerical vars and encoded categorical vars.

df_final_merge=pd.concat([df_merged_num,df_dummy_cat],axis=1)
# the above final_merge contains both train & test data(remember we combined both), time to separate them now

df_train_final = df_final_merge.iloc[0:1438, :] # first 1438 rows were train data

df_test_final = df_final_merge.iloc[1438:, :]   #all rows below 1438 were test data

print(df_train_final.shape)

print(df_test_final.shape)

print(y_train.shape)      #our target from train data we separated