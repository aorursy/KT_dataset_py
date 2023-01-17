#using in C# is import in Python

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestRegressor

from scipy import stats

from scipy.stats import norm



#These 2 libraries are used to plot graphs

import seaborn as sns

import matplotlib.pyplot as plt



#SimnpleImputer will be used for feature Engineering

from sklearn.impute import SimpleImputer
#pd is alias for pandas library

train_raw = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv') 

test_raw = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

train = train_raw.copy()

test = test_raw.copy()

print("Shape -> Train :", train.shape, "Test :", test.shape)
#Skew & Kurt Before

print("Before -> Skew :", train.SalePrice.skew(), "Kurt :", train.SalePrice.kurt())



#plotting distribution of SalePrice

fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(20,5))

plt.subplot(1,2,1) #args->row,col,index

plt.xlabel('SalePrice')

plt.ylabel('Frequency')

plt.title('SalePrice Distribution')

sns.distplot(train.SalePrice,fit=norm)

plt.subplot(1,2,2)

stats.probplot(train['SalePrice'], plot=plt)

plt.ylabel('SalePrice')

plt.title('SalePrice Probility Plot')

plt.show()


#apply log to SalePrice

train.SalePrice = np.log1p(train.SalePrice)



#Skew & Kurt After

print("After -> Skew :", train.SalePrice.skew(), "Kurt :", train.SalePrice.kurt())



#plotting distribution of SalePrice

fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(20,5))

plt.subplot(1,2,1) #args->row,col,index

plt.xlabel('SalePrice')

plt.ylabel('Frequency')

plt.title('SalePrice Distribution')

sns.distplot(train.SalePrice,fit=norm)

plt.subplot(1,2,2)

stats.probplot(train['SalePrice'], plot=plt)

plt.ylabel('SalePrice')

plt.title('SalePrice Probility Plot')

plt.show()

#droping Id column

train.drop('Id',axis=1,inplace=True)

test.drop('Id',axis=1,inplace=True)

print("Shape -> Train :", train.shape, "Test :", test.shape)
#Keep Only Numeric Columns

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

train = train.select_dtypes(include=numerics)

numericColList = train.columns.tolist()

numericColList.remove('SalePrice')

test = test[numericColList]

print("Shape -> Train :", train.shape, "Test :", test.shape)
#Plot Heat Map to find relationship

corrmat = train.corr()

fig, ax = plt.subplots(figsize=(12,9))

sns.heatmap(corrmat,vmax=0.8,square=True)

plt.show()
#SalePrice correlation matrix

k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

fig, ax = plt.subplots(figsize=(12,9))

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
#Columns Selected (removed columns which can be derived)

selectedCols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

train = train[selectedCols]

selectedCols.remove('SalePrice')

test = test[selectedCols]
#Removing Columns missing data in 

print("Training Missing before : " ,train.isnull().sum().sum())

print("Test Missing before : " ,test.isnull().sum().sum())

test_na = test.isnull().sum()/len(train)*100

test_na = test_na.drop(test_na[test_na == 0].index).sort_values(ascending=False)

missing_data = pd.DataFrame({'Missing Ratio' : test_na})

#Print Missing Data

print(missing_data)

#Rows missing Data

print(test[test.isnull().any(axis=1)])

#replacing missing data with 0

test.fillna(0,inplace=True)

print("Training Missing after : " ,train.isnull().sum().sum())

print("Test Missing after : " ,test.isnull().sum().sum())
train_x = train.loc[:,train.columns != 'SalePrice']

train_y = train.SalePrice

test_x = test

train_x.shape, train_y.shape, test_x.shape
#Regression

model = RandomForestRegressor()

model.fit(train_x,train_y)

pred_y = model.predict(test_x)

pred_y = np.expm1(pred_y)
#Submission

my_submission = pd.DataFrame({'Id':test_raw.Id, 'SalePrice' : pred_y})

my_submission.to_csv('submission.csv', index=False)