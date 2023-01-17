# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk(r'../input/home-data-for-ml-course/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train=pd.read_csv(r"../input/home-data-for-ml-course/train.csv")

test=pd.read_csv(r"../input/home-data-for-ml-course/train.csv")



#Percentage of missing values in each column

train.isnull().sum()/train.shape[0]

#remove variables with more than 50% missing data

#PoolQC

#MiscFeature

#Alley

#Fence

#FireplaceQu
train=train.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu'],axis=1)

test=test.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu'],axis=1)

train['SalePrice'].describe()
corrmatrix=train.corr()

f,ax=plt.subplots(figsize=(12,9))

sns.heatmap(corrmatrix,vmax=.8,square=True)
corrmatrix_impvar=pd.DataFrame(corrmatrix.iloc[-1,:])

# Considering all the variables where correlation is greater than 0.4

corrmatrix_impvar=corrmatrix_impvar[corrmatrix_impvar['SalePrice']>0.4]

multicorr=train[list(corrmatrix_impvar.index)]

multicorr_matrix=multicorr.corr()

print(multicorr_matrix)
catvars=train.select_dtypes('object')

catvars.isnull().sum()/catvars.shape[0]
#filling missing values for all categorical values with their mode

catvars=catvars.apply(lambda x: x.fillna(x.mode()[0]),axis=1)
#Label encode the categorical variables

le=LabelEncoder()

catvars=catvars.apply(le.fit_transform)

catvars=pd.merge(catvars,train['SalePrice'],left_index=True,right_index=True)

catvars_corr=catvars.corr()

catvars_corr=pd.DataFrame(catvars_corr.iloc[-1,:])

#Select all variable whose correlation with the target variable(SalePrice)>0.4 or <-0.4

catvars_corr=catvars_corr[(catvars_corr['SalePrice']<-0.4) | (catvars_corr['SalePrice']>0.4)]

train_cat=catvars[list(catvars_corr.index)]

train_corr=train_cat.corr()

print(train_corr)
train=train.loc[:,['YearBuilt','TotalBsmtSF','GrLivArea',

            'GarageArea','OverallQual','MasVnrArea','Fireplaces','ExterQual',

            'GarageType','HeatingQC','SalePrice']]



train['ExterQual']=le.fit_transform(train['ExterQual'])

train['GarageType']=train['GarageType'].fillna(train['GarageType'].mode()[0])

train['GarageType']=le.fit_transform(train['GarageType'])

train['HeatingQC']=le.fit_transform(train['HeatingQC'])



final_corr_train=train.corr()

print(final_corr_train)
g=sns.PairGrid(train)

g.map(plt.scatter)

sns.relplot('TotalBsmtSF','SalePrice',kind='scatter',data=train)

##There are houses with TotalBsmtSF=0; 

sns.relplot('GrLivArea','SalePrice',kind='scatter',data=train)

##linear correlation

sns.relplot('GarageArea','SalePrice',kind='scatter',data=train)

##There are houses with GarageArea=0;

sns.relplot('OverallQual','SalePrice',kind='scatter',data=train)

##SalePrice increases with overall quality
sns.relplot('MasVnrArea','SalePrice',kind='scatter',data=train)
sns.relplot('Fireplaces','SalePrice',kind='scatter',data=train)

sns.relplot('ExterQual','SalePrice',kind='scatter',data=train)

sns.relplot('GarageType','SalePrice',kind='scatter',data=train)

sns.relplot('HeatingQC','SalePrice',kind='scatter',data=train)
#checking for missing values in selected variables

train.isnull().sum()
#Imputing values for MasVnrArea variable

train['MasVnrArea']=[train['MasVnrArea'].mean() if(pd.isnull(x)) else x for x in train['MasVnrArea']]

train.isnull().sum()