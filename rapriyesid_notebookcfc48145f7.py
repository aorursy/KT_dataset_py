import sys

from time import time

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
loc_train="../input/train.csv"

loc_test="../input/test.csv"

train=pd.read_csv(loc_train)

test=pd.read_csv(loc_test)

target=train.SalePrice





def combine_data():

    global combined

    

    print (train.shape,test.shape)

    

    

    #Dropping SalePrice

    train.drop(['SalePrice'],axis=1,inplace=True)

    print( train.shape)

    #Combining Data

    combined=train.append(test)

    combined.reset_index(inplace=True)

    combined.drop(['index'],1,inplace=True)

    print (combined.shape)

    return combined



combine_data() 
# No Id

# No MSSubClass

# No MSZoning

# Yes LotFrontage

# Yes LotArea

# Yes Street

# No Alley

# No LotShape

# No LandContour

# No Utilities

# No LotConfig

# No LandSlope

# Yes Neighborhood

# Yes Condition1

# Yes Condition2

# No BldgType

# No HouseStyle

# OverallQual

# OverallCond

# YearBuilt

# YearRemodAdd

# RoofStyle

# RoofMatl

# Exterior1st

# Exterior2nd

# MasVnrType

# MasVnrArea

# ExterQual

# ExterCond

# Foundation

# BsmtQual

# BsmtCond

# BsmtExposure

# BsmtFinType1

# BsmtFinSF1

# BsmtFinType2

# BsmtFinSF2

# BsmtUnfSF

# TotalBsmtSF

# Heating

# HeatingQC

# CentralAir

# Electrical

# 1stFlrSF

# 2ndFlrSF

# LowQualFinSF

# GrLivArea

# BsmtFullBath

# BsmtHalfBath

# FullBath

# HalfBath

# BedroomAbvGr

# KitchenAbvGr

# KitchenQual

# TotRmsAbvGrd

# Functional

# Fireplaces

# FireplaceQu

# GarageType

# GarageYrBlt

# GarageFinish

# GarageCars

# GarageArea

# GarageQual

# GarageCond

# PavedDrive

# WoodDeckSF

# OpenPorchSF

# EnclosedPorch

# 3SsnPorch

# ScreenPorch

# PoolArea

# PoolQC

# Fence

# MiscFeature

# MiscVal

# MoSold

# YrSold

# SaleType

# SaleCondition
var = 'BsmtFinType1'

plt.subplots(figsize=(20, 13))

fig = sns.boxplot(x=var, y=target, data=train)

fig.axis(ymin=0, ymax=800000);
#Plotting Data. All columns together



#for i in train.describe().columns:

#    val=i

#    plt.subplots(figsize=(20, 13))

#    plt.scatter(train[val],target)

#    plt.show()

#    print (i)

#    print ("---------------")



#------BOXPLOT------



#var = 'MSZoning'

#plt.subplots(figsize=(20, 13))

#fig = sns.boxplot(x=var, y="SalePrice", data=train)

#fig.axis(ymin=0, ymax=800000);
#Retaining only Useful columns(Linear Relation between 'Var' and 'SalePrice')

#MSSubClass No#LotFrontage Yes#LotArea Yes#OverallQual Yes#OverallCond Yes#YearBuilt YN#YearRemodAdd YN#MasVnrArea YN#BsmtFinSF1 Yes#TotalBsmtSF#1stFlrSF#2ndFlrSF#LowQualFinSF

#OpenPorchSF#WoodDeckSF#GarageArea#GarageCars#GarageYrBlt

#Fireplaces#TotRmsAbvGrd#KitchenAbvGr#GrLivArea



#df=pd.DataFrame()

#for i in combined.describe().columns:

#    df[i]=combined[i]

#df.describe()   



#-------OR----------



df=pd.DataFrame()

#df['Id']=combined['Id']

#df['']=combined['']





df['Street']=combined['Street']

df['Neighborhood']=combined['Neighborhood']

df['LotShape']=combined['LotShape']

df['Condition2']=combined['Condition2']

df['Condition1']=combined['Condition1']

df['HouseStyle']=combined['HouseStyle']

df['BsmtFinType1']=combined['BsmtFinType1']

#df['']=combined['']

#df['']=combined['']

#df['']=combined['']

df['LotFrontage']=combined['LotFrontage']

df['LotArea']=combined['LotArea']

df['OverallQual']=combined['OverallQual']

df['OverallCond']=combined['OverallCond']

df['BsmtFinSF1']=combined['BsmtFinSF1']

df['TotalBsmtSF']=combined['TotalBsmtSF']

df['1stFlrSF']=combined['1stFlrSF']

df['2ndFlrSF']=combined['2ndFlrSF']

df['LowQualFinSF']=combined['LowQualFinSF']

df['OpenPorchSF']=combined['OpenPorchSF']

df['WoodDeckSF']=combined['WoodDeckSF']

df['GarageArea']=combined['GarageArea']

df['GarageCars']=combined['GarageCars']

df['GarageYrBlt']=combined['GarageYrBlt']

df['Fireplaces']=combined['Fireplaces']

df['TotRmsAbvGrd']=combined['TotRmsAbvGrd']

df['KitchenAbvGr']=combined['KitchenAbvGr']

df['GrLivArea']=combined['GrLivArea']



df.describe()
titles_dummies = pd.get_dummies(df['Neighborhood'],prefix='Neighborhood')

df = pd.concat([df,titles_dummies],axis=1)

df.drop('Neighborhood',axis=1,inplace=True)

    

titles_dummies = pd.get_dummies(df['LotShape'],prefix='LotShape')

df = pd.concat([df,titles_dummies],axis=1)

df.drop('LotShape',axis=1,inplace=True)



titles_dummies = pd.get_dummies(df['Condition2'],prefix='Condition2')

df = pd.concat([df,titles_dummies],axis=1)

df.drop('Condition2',axis=1,inplace=True)



titles_dummies = pd.get_dummies(df['Condition1'],prefix='Condition1')

df = pd.concat([df,titles_dummies],axis=1)

df.drop('Condition1',axis=1,inplace=True)



titles_dummies = pd.get_dummies(df['HouseStyle'],prefix='HouseStyle')

df = pd.concat([df,titles_dummies],axis=1)

df.drop('HouseStyle',axis=1,inplace=True)

    

titles_dummies = pd.get_dummies(df['BsmtFinType1'],prefix='BsmtFinType1')

df = pd.concat([df,titles_dummies],axis=1)

df.drop('BsmtFinType1',axis=1,inplace=True)



    
#Filling Missing Values, Based on columns Median()



#Finding Missing values

#for i in df.columns:

#    a=df[i].isnull().value_counts()

#    print (i)

#   print (a)

#    print ("------")

    

df['GarageCars'].fillna(df['GarageCars'].median(),inplace=True)

df['GarageYrBlt'].fillna(df['GarageYrBlt'].median(),inplace=True)

df['GarageCars'].fillna(df['GarageCars'].median(),inplace=True)

df['GarageArea'].fillna(df['GarageArea'].median(),inplace=True)

df['TotalBsmtSF'].fillna(df['TotalBsmtSF'].median(),inplace=True)

df['BsmtFinSF1'].fillna(df['BsmtFinSF1'].median(),inplace=True)

df['LotFrontage'].fillna(df['LotFrontage'].median(),inplace=True)

df['Street']=df['Street'].map({'Pave':1,'Grvl':0})



df.describe()

#Seperating Test and Train



train_new=df[0:1460]

test_new=df[1460:]



print (train_new.shape)

print (test_new.shape)

print (target.shape)
clf=RandomForestClassifier(n_estimators=1000)

t0=time()

clf.fit(train_new,target)

print (round(time()-t0,3))

t1=time()

pred=clf.predict(test_new)

print (round(time()-t1,3))
output=pd.DataFrame()

output['Id']=test['Id']

output['SalePrice']=pred



output[['Id','SalePrice']].to_csv("output.csv",index=False)
